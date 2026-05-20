use anyhow::Context;
use crate::{
    Config,
    // Constants
    RAW_W, RAW_H,
    BYTES_PER_RAW_FRAME,
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

/// Stream to the egui GUI for camera aiming and focus selection
pub fn gui_stream(mut user_conf: Config, stream_tx: crossbeam_channel::Sender<Vec<u8>>, rx_f: Receiver<f32>) {
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
    cfg.set_size(Size { width: 1536, height: 864 });
    
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
    loop {
        // Check the channel for a message, timeout after 2 seconds
        let mut req = rx.recv_timeout(Duration::from_secs(2)).expect("Camera request failed");

        // Get framebuffer for the stream
        let framebuffer: &MemoryMappedFrameBuffer<FrameBuffer> = req.buffer(&stream).unwrap();

        let planes: Vec<&[u8]> = framebuffer.data();
        let mut raw_planes: Vec<u8> = Vec::with_capacity(BYTES_PER_RAW_FRAME);
        for p in &planes {
            raw_planes.extend_from_slice(p);
        }

        let _ = stream_tx.try_send(raw_planes);

        //println!("{:#?}", req.metadata());
        req.reuse(ReuseFlag::REUSE_BUFFERS);

        // Check for user-specified focus
        let focus_val = rx_f.try_recv();
        if let Ok(f) = focus_val {
            user_conf.set_focus(f);
        }
        let focus_val: f32 = user_conf.focus * 16.0;
        let current_controls = req.controls_mut();

        current_controls.set(controls::LensPosition(focus_val)).expect("Couldn't set focus");

        cam.queue_request(req).unwrap();
    }
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
        .args(["-framerate", "120"]) // Remember to change this when setting framerate!
        .args(["-s", format!("{}x{}", RAW_W, RAW_H).as_str()])
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

        let planes: Vec<&[u8]> = framebuffer.data();
        let contiguous_planes: Vec<u8> = planes[..].concat();

        // Send over the pipe to ffmpeg
        writer.write_all(&contiguous_planes).expect("Couldn't write frame to ffmpeg pipe");


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
    globals.set(controls::AfMode::Manual).unwrap();
    globals.set(controls::AfRange::Macro).unwrap();

    let focus_val: f32 = _user_conf.focus * 16.0;
    globals.set(controls::LensPosition(focus_val)).unwrap();

    globals
}
