use anyhow::Context;
use crate::{
    Config,
    // Constants
    RAW_W, RAW_H,
    BYTES_PER_RAW_FRAME,
    FIRST_CROP_W, FIRST_CROP_H, // intermediate crop size for GUI
    TRAINING_CROP_W, TRAINING_CROP_H, // crop with padding for training wiggle
};
use crossbeam_channel;
use libcamera::{
    camera::{ActiveCamera, CameraConfigurationStatus},
    camera_manager::CameraManager, 
    control::ControlList,
    controls,
    framebuffer_allocator::{FrameBuffer, FrameBufferAllocator},
    framebuffer_map::MemoryMappedFrameBuffer,
    geometry::Size,
    pixel_format::PixelFormat,
    properties::Model,
    request::{Request, RequestStatus, ReuseFlag},
    stream::StreamRole,
    utils::UniquePtr,
};
use std::{
    fs::OpenOptions,
    io::{Write, pipe},
    process::Command,
    sync::mpsc::{
        channel, 
        Receiver,
    },
    time::{
        Duration,
        Instant,
        SystemTime,
    },
};

// Define MJPEG format. From the libcamera-rs examples: drm-fourcc doesn't include
// an enum variant for MJPEG, so we construct it manually from the raw fourcc identifier
const PIXEL_FORMAT_YU12: PixelFormat = PixelFormat::new(u32::from_le_bytes([b'Y', b'U', b'1', b'2']), 0);

/// Stream to the egui GUI for ROI selection
pub fn gui_stream(user_conf: Config, stream_tx: crossbeam_channel::Sender<Vec<u8>>, rx_r: Receiver<(u32, u32)>) -> (u32, u32) {
    // Interface for choosing a camera
    let cm = CameraManager::new().unwrap();

    // CameraList<'_>
    let cams = cm.cameras();

    // Get the first Camera<'_> from the list
    let cam = cams.get(0).expect("no camera 0");

    println!(
        "\nUsing camera: {}",
        *cam.properties().get::<Model>().unwrap()
    );

    // Activate the first camera in the list (becomes an ActiveCamera<'_>)
    let mut cam: ActiveCamera = cam.acquire().expect("Could not acquire camera");

    // -- CONFIGURE --
    //
    // We don't need to do anything with the return except panic! if it fails
    // &[StreamRole::<role>] generates the default config for the given role
    let mut cfgs = cam.generate_configuration(&[StreamRole::VideoRecording]).unwrap();

    // Set pixel format
    cfgs.get_mut(0).unwrap().set_pixel_format(PIXEL_FORMAT_YU12);

    // Can set capture size here (but it will just downsample unless a supported
    // capture resolution/format is chosen).
    // The supported resolutions for the IMX708 are:
    //  - 2304x1296 @ 56  Hz
    //  - 2304x1296 @ 30  Hz
    //  - 1536x864  @ 120 Hz
    let cfg = &mut cfgs.get_mut(0).unwrap();
    cfg.set_size(Size { width: 2304, height: 1296 });
    
    // Validate config
    match cfgs.validate() {
        CameraConfigurationStatus::Valid => println!("Camera config valid.\n"),
        CameraConfigurationStatus::Adjusted => println!("Camera config valid after adjustments: {cfgs:#?}\n"),
        CameraConfigurationStatus::Invalid => panic!("Error validating configuration\n"),
    }
    cam.configure(&mut cfgs).expect("Unable to configure camera");

    // -- ALLOCATE MEMORY --
    //
    // This struct allocates contiguous memory for saving frames. It needs to know
    // at runtime how much memory to allocate. Calling alloc.alloc(cam) uses the
    // Stream underlying the ActiveCamera to determine the amount of memory needed.
    let mut alloc = FrameBufferAllocator::new(&cam);

    // Get a ref to the StreamConfigurationRef underlying the
    // CameraConfiguration, then access the underlying Stream object
    let cfg = &cfgs.get(0).unwrap();
    let stream = cfg.stream().unwrap();

    // Determine stride of Y plane (should be padded a bit from 2028)
    let y_stride: u32 = cfg.get_stride();

    // Allocate memory for the necessary FrameBuffers
    let bufs = alloc.alloc(&stream).unwrap();

    // Convert FrameBuffer to MemoryMappedFrameBuffer (which apparently is necessary for
    // reading slices of bytes) and collect into a Vector
    let bufs = bufs
        .into_iter()
        .map(|buf| MemoryMappedFrameBuffer::new(buf).unwrap())
        .collect::<Vec<_>>();

    // -- FINALIZE CONFIGURATION --
    //
    // Check the user-specified config for controls
    // that can be set globally (not per request)
    let globals = global_config(&user_conf);

    // Start the camera (finalize configuration and permit queue_request() calls)
    // We pass an Option<ControlList> as the parameter
    cam.start(Some(&globals)).unwrap();

    // Create the capture requests and attach them to buffers, then collect
    // into a Vector
    let reqs = bufs
        .into_iter()
        .enumerate()
        .map(|(i, buf)| {
            let mut req = cam.create_request(Some(i as u64)).unwrap();
            req.add_buffer(&stream, buf).unwrap();
            req
        })
        .collect::<Vec<_>>();

    // Multiple producer single consumer channel for communication with the recording thread
    let (tx, rx) = channel();

    // Callback executed when frame is captured
    cam.on_request_completed(move |req: Request| {
        tx.send(req).unwrap();
    });

    // Add all requests to the queue
    for req in reqs {
        cam.queue_request(req).unwrap();
    }
    // Main loop, loops until user interrupt
    let crop_x: u32;
    let crop_y: u32;
    loop {
        // First, check if the user selected the ROI yet (try_recv so we don't block)
        if let Ok((x,y)) = rx_r.try_recv() {
            println!("ROI selection received: x: {x}, y: {y}");       
            crop_x = x;
            crop_y = y;
            break;
        }
        // Check the channel for a message, timeout after 2 seconds
        let mut req = rx.recv_timeout(Duration::from_secs(2)).expect("Camera request failed");

        // Get framebuffer for the stream
        let framebuffer: &MemoryMappedFrameBuffer<FrameBuffer> = req.buffer(&stream).unwrap();

        let planes: Vec<&[u8]> = framebuffer.data();
        let mut raw_planes: Vec<u8> = Vec::with_capacity(BYTES_PER_RAW_FRAME);
        for p in &planes {
            raw_planes.extend_from_slice(p);
        }
        let cropped_planes: Vec<u8> = crop_frame(planes, y_stride as usize, &user_conf);

        let _ = stream_tx.try_send(cropped_planes);

        req.reuse(ReuseFlag::REUSE_BUFFERS);
        cam.queue_request(req).unwrap();
    }
    (crop_x, crop_y)
}

/// Set camera configuration based on user_conf, allocate memory,
/// and record.
pub fn record(user_conf: &Config) {
    // Interface for choosing a camera
    let cm = CameraManager::new().unwrap();

    // CameraList<'_>
    let cams = cm.cameras();

    // Get the first Camera<'_> from the list
    let cam = cams.get(0).expect("no camera 0");

    println!(
        "\nUsing camera: {}",
        *cam.properties().get::<Model>().unwrap()
    );

    // Activate the first camera in the list (becomes an ActiveCamera<'_>)
    let mut cam: ActiveCamera = cam.acquire().expect("Could not acquire camera");

    // -- CONFIGURE --
    //
    // We don't need to do anything with the return except panic! if it fails
    // &[StreamRole::<role>] generates the default config for the given role
    let mut cfgs = cam.generate_configuration(&[StreamRole::VideoRecording]).unwrap();

    // Set pixel format
    cfgs.get_mut(0).unwrap().set_pixel_format(PIXEL_FORMAT_YU12);

    // Can set capture size here (but it will just downsample unless a supported
    // capture resolution/format is chosen).
    // The supported resolutions for the IMX708 are:
    //  - 2304x1296 @ 56  Hz
    //  - 2304x1296 @ 30  Hz
    //  - 1536x864  @ 120 Hz
    let cfg = &mut cfgs.get_mut(0).unwrap();
    cfg.set_size(Size { width: 2304, height: 1296 });
    
    // Validate config
    match cfgs.validate() {
        CameraConfigurationStatus::Valid => println!("Camera config valid.\n"),
        CameraConfigurationStatus::Adjusted => println!("Camera config valid after adjustments: {cfgs:#?}\n"),
        CameraConfigurationStatus::Invalid => panic!("Error validating configuration\n"),
    }
    cam.configure(&mut cfgs).expect("Unable to configure camera");

    // -- ALLOCATE MEMORY --
    //
    // This struct allocates contiguous memory for saving frames. It needs to know
    // at runtime how much memory to allocate. Calling alloc.alloc(cam) uses the
    // Stream underlying the ActiveCamera to determine the amount of memory needed.
    let mut alloc = FrameBufferAllocator::new(&cam);

    // Get a ref to the StreamConfigurationRef underlying the
    // CameraConfiguration, then access the underlying Stream object
    let cfg = &cfgs.get(0).unwrap();
    let stream = cfg.stream().unwrap();

    // Determine stride of Y plane (should be padded a bit from 2028)
    let y_stride: u32 = cfg.get_stride();

    // Allocate memory for the necessary FrameBuffers
    let bufs = alloc.alloc(&stream).unwrap();

    // Convert FrameBuffer to MemoryMappedFrameBuffer (which apparently is necessary for
    // reading slices of bytes) and collect into a Vector
    let bufs = bufs
        .into_iter()
        .map(|buf| MemoryMappedFrameBuffer::new(buf).unwrap())
        .collect::<Vec<_>>();

    // -- FINALIZE CONFIGURATION --
    //
    // Check the user-specified config for controls
    // that can be set globally (not per request)
    let globals = global_config(user_conf);

    // Start the camera (finalize configuration and permit queue_request() calls)
    // We pass an Option<ControlList> as the parameter
    cam.start(Some(&globals)).unwrap();

    // Create the capture requests and attach them to buffers, then collect
    // into a Vector
    let reqs = bufs
        .into_iter()
        .enumerate()
        .map(|(i, buf)| {
            let mut req = cam.create_request(Some(i as u64)).unwrap();
            req.add_buffer(&stream, buf).unwrap();
            req
        })
        .collect::<Vec<_>>();

    // Multiple producer single consumer channel for communication with the recording thread
    let (tx, rx) = channel();

    // Callback executed when frame is captured
    cam.on_request_completed(move |req: Request| {
        if req.status() == RequestStatus::Complete {
            // Where we would have polled the IMX500 NPU results
        }
        tx.send(req).unwrap();
    });

    // Add all requests to the queue
    for req in reqs {
        cam.queue_request(req).unwrap();
    }

    // Open pipe for ffmpeg conversion
    let (reader, mut writer) = pipe().expect("Couldn't open pipe to ffmpeg");

    // Start ffmpeg
    let mut ffmpeg_cmd = Command::new("ffmpeg");
    ffmpeg_cmd
        .args(["-pix_fmt", "yuv420p"])
        .args(["-f", "rawvideo"])
        .args(["-framerate", "30"])
        // using training crop for now when saving
        .args(["-s", format!("{}x{}", TRAINING_CROP_W, TRAINING_CROP_H).as_str()])
        .args(["-i", "pipe:0"])
        .arg(&user_conf.filename)
        .stdin(reader); // pass the read end of the pipe

    ffmpeg_cmd.spawn().expect("Couldn't start ffmpeg thread");

    // Open file for writing frame timestamps
    let timestamp_filename = user_conf.filename.clone();
    let timestamp_filename = format!("{}.txt", timestamp_filename.split('.').next().unwrap());
    
    let mut timestamp_file = OpenOptions::new()
        .read(false)
        .write(true)
        .create(true)
        .truncate(true)
        .open(timestamp_filename)
        .context("open timestamp file w")
        .expect("oops (timestamp file creation)");
    // Check exactly when we are starting in nanoseconds since 1/1/1970
    let system_start_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)
        .expect("System time is before 1/1/1970");

    timestamp_file.write_all(format!("System time recorded at capture start (ns since 1/1/1970): {}\n", system_start_time.as_nanos()).as_bytes()).expect("Couldn't write initial line to timestamp file");
    timestamp_file.write_all("Remaining lines each contain #ns since start time\n".as_bytes()).expect("Couldn't write second line to timestamp file");

    // Need to also store an Instant that we are starting, so we can use elapsed() to check the
    // diff
    let start_time = Instant::now();
    // Main loop, loops until user interrupt
    loop {
        // Check the channel for a message, timeout after 2 seconds
        let mut req = rx.recv_timeout(Duration::from_secs(2)).expect("Camera request failed");

        // Write the current time elapsed for this frame
        // I think we should do it before the cropping and writing so that it's as
        // close to the time when the photons hit the sensor as possible
        let loop_time = start_time.elapsed();
        let loop_time = format!("{}\n", loop_time.as_nanos());
        let _ = timestamp_file.write_all(loop_time.as_bytes());

        // Get framebuffer for the stream
        let framebuffer: &MemoryMappedFrameBuffer<FrameBuffer> = req.buffer(&stream).unwrap();

        // Pull out the data and crop the frame
        let planes: Vec<&[u8]> = framebuffer.data();
        let cropped_planes: Vec<u8> = crop_frame(planes, y_stride as usize, user_conf);

        // Send over the pipe to ffmpeg
        writer.write_all(&cropped_planes[..]).expect("Couldn't write frame to ffmpeg pipe");

        // Reuse the buffers so we don't have to reallocate every frame
        req.reuse(ReuseFlag::REUSE_BUFFERS);
        cam.queue_request(req).unwrap();
    }
}

/// Returns a UniquePtr<ControlList> (pretty sure this is an abstraction of a concept from C)
/// with the user-specified configs set
fn global_config(_user_conf: &Config) -> UniquePtr<ControlList> {
    let mut globals = ControlList::new();

    let target_fps = 56.0;
    let frame_duration = (1_000_000.0 / target_fps) as i64;

    globals.set(controls::FrameDurationLimits([frame_duration, frame_duration])).unwrap();
    globals.set(controls::AfMode::Continuous).unwrap();
    globals.set(controls::AfRange::Macro).unwrap();

    globals
}

/// Handles frame cropping (for saving or streaming). Needs y_stride from the original Y plane,
/// so that we can account for the zero padding on left/right that the camera does
fn crop_frame(planes: Vec<&[u8]>, y_stride: usize, user_conf: &Config) -> Vec<u8> {
    let w: usize;
    let h: usize;
    let mut x: usize;
    let mut y: usize;
    if user_conf.roi_selected {
        // At least for now, we save with 100 px padding around the edges for 
        // use in training
        w = TRAINING_CROP_W as usize;
        h = TRAINING_CROP_H as usize;
        // Make sure the user-defined crop is an even number of pixels in both axes
        x = (user_conf.crop_x & !1) as usize;
        y = (user_conf.crop_y & !1) as usize;

        // During the training crop, we need to adjust x and y to account for the
        // padding. 100 px => decrease x and y by 50 px each. We also need to account
        // for the pixels missing due to the intermediate crop, so we end up with
        // for example:
        // x: 2304 - 480 = 1980 / 2 = 990 - 50 = 940
        // y: 1296 - 480 = 972 / 2 = 486 - 50 = 436
        // Note that we still compute the x_tmp and y_tmp from the
        // origin of the FIRST_CROP, rather than TRAINING_CROP
        let x_tmp = (((RAW_W - FIRST_CROP_W) / 2) - 50) as usize;
        x += x_tmp;
        let y_tmp = (((RAW_H - FIRST_CROP_H) / 2) - 50) as usize;
        y += y_tmp;
    } else { // if we haven't selected the ROI yet, don't crop as far
        w = FIRST_CROP_W as usize;
        h = FIRST_CROP_H as usize;
        // If we haven't set the ROI yet, default to 480x480 in the center of the frame
        let x_tmp = ((RAW_W - FIRST_CROP_W) / 2) as usize;
        x = x_tmp;
        let y_tmp = ((RAW_H - FIRST_CROP_H) / 2) as usize;
        y = y_tmp;
        println!("x:{}, y:{}", x, y);
    }
    // planes contains Y, U, and V planes. Y is double the height/width and stride
    let y_plane = planes[0];
    let u_plane = planes[1];
    let v_plane = planes[2];


    // Define the UV coords (half the size)
    let uvx: usize = x / 2;
    let uvy: usize = y / 2;
    let uvw: usize = w / 2;
    let uvh: usize = h / 2;

    // Length of each row in the planes
    let uv_stride: usize = y_stride / 2;

    // Determine new plane sizes
    let y_size: usize = w * h;
    let uv_size: usize = uvw * uvh;

    // Output vector
    let mut out: Vec<u8> = Vec::with_capacity(y_size + 2 * uv_size);

    // Crop Y
    for row in 0..h {
        let row_start_idx = (y + row) * y_stride + x;
        out.extend_from_slice(&y_plane[row_start_idx..row_start_idx + w]);
    }

    // Crop U & V
    for row in 0..uvh {
        let row_start_idx = (uvy + row) * uv_stride + uvx;
        out.extend_from_slice(&u_plane[row_start_idx..row_start_idx + uvw]);
    }
    for row in 0..uvh {
        let row_start_idx = (uvy + row) * uv_stride + uvx;
        out.extend_from_slice(&v_plane[row_start_idx..row_start_idx + uvw]);
    }
    out
}
