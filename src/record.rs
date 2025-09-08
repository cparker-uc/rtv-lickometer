use anyhow::Context;
use crate::{
    Config,
    // Constants
    RAW_W, RAW_H, BYTES_PER_RAW_FRAME,
    BYTES_PER_RAW_Y_PLANE, BYTES_PER_RAW_UV_PLANE,
    CROP_W, CROP_H, BYTES_PER_CROPPED_FRAME,
    FIRST_CROP_W, FIRST_CROP_H, BYTES_PER_FIRST_CROP_FRAME,
};
use crossbeam_channel;
use libcamera::{
    camera::{ActiveCamera, CameraConfigurationStatus},
    camera_manager::CameraManager, 
    control::ControlList,
    control_value::ControlValue,
    controls,
    framebuffer::AsFrameBuffer,
    framebuffer_allocator::{FrameBuffer, FrameBufferAllocator},
    framebuffer_map::MemoryMappedFrameBuffer,
    geometry::{Rectangle, Size},
    pixel_format::PixelFormat,
    properties::Model,
    request::{Request, RequestStatus, ReuseFlag},
    stream::{StreamRole, Stream},
    utils::UniquePtr,
};
use libcamera_sys;
use std::{
    env::Args,
    error::Error,
    fs::{self, OpenOptions, File},
    io::{Read, Write, pipe, prelude::*},
    mem::zeroed,
    os::fd::AsRawFd,
    path::Path,
    process::Command,
    sync::mpsc::{
        channel, 
        Sender,
        Receiver,
    },
    time::Duration,
};
use v4l::{
    Device, Control, control::Value,
};
use v4l2_sys_mit as v4l2;

// Low-level libcamera-sys representations of the IMX500 specific metadata
const CNN_OUTPUT_ID: u32 = libcamera_sys::CNN_OUTPUT_TENSOR as u32;
const CNN_OUTPUT_INFO_ID: u32 = libcamera_sys::CNN_OUTPUT_TENSOR_INFO as u32;
const CNN_INPUT_ID: u32 = libcamera_sys::CNN_INPUT_TENSOR as u32;
const CNN_INPUT_INFO_ID: u32 = libcamera_sys::CNN_INPUT_TENSOR_INFO as u32;
const CNN_ENABLE_INPUT_TENSOR_ID: u32 = libcamera_sys::CNN_ENABLE_INPUT_TENSOR as u32;
const CNN_KPI_INFO_ID: u32 = libcamera_sys::CNN_KPI_INFO as u32;

// Define MJPEG format. From the libcamera-rs examples: drm-fourcc doesn't include
// an enum variant for MJPEG, so we construct it manually from the raw fourcc identifier
const PIXEL_FORMAT_YU12: PixelFormat = PixelFormat::new(u32::from_le_bytes([b'Y', b'U', b'1', b'2']), 0);

/// Open the IMX500 with V4L and write the CNN file to the NPU
pub fn load_firmware(user_conf: &Config) -> Result<(), Box<dyn Error>> {
    // Loop through the /dev directory and check the devices
    let mut dev_dir = fs::read_dir(&Path::new("/dev/"))?;
    let mut subdev_num: usize = 0;
    while let Some(Ok(d)) = dev_dir.next() {
        // If the current entry has v4l-subdev in the path,
        // check it for IMX500 controls
        let d_path = d.path();
        let mut d_str = d_path.to_str().unwrap();
        if let Some(_) = d_str.find("v4l-subdev") {
            let dev = Device::with_path(&d.path()).expect("couldn't open device");
            if let Ok(_) = dev.query_controls() {
                let split_path = d_str.split("subdev");
                let num = split_path.last().unwrap();
                subdev_num = num.parse()?;
            }
        }
    }
    let subdev_path = format!("/dev/v4l-subdev{}", subdev_num);
    println!("Opening {} to load firmware", subdev_path);
    let imx500 = Device::with_path(&Path::new(&subdev_path))
                              .expect("Failed to open device");
    let ctrls = imx500.query_controls()?;

    // Get the correct control based on the name
    let Some(ctrl) = ctrls.iter().find(|item| item.name == "IMX500 Network Firmware File FD") else {
        panic!("No IMX500 firmware file control found");
    };

    // Grab the CNN file
    // TODO: Make this a CLI option
    let rpk_file: File = OpenOptions::new()
        .read(true)
        .open("/usr/share/imx500-models/imx500_network_inputtensoronly.rpk")
        .expect("Couldn't open requested CNN file");
    let rpk_fd: i32 = rpk_file.as_raw_fd();

    let rpk_fd: Value = Value::Integer(rpk_fd.into());
    let ctrl = Control { id: ctrl.id, value: rpk_fd };
    imx500.set_control(ctrl).unwrap();

    Ok(())
}

/// Stream to the egui GUI for ROI selection
pub fn gui_stream(user_conf: &mut Config, stream_tx: crossbeam_channel::Sender<Vec<u8>>, rx_r: Receiver<(u32, u32)>) -> (u32, u32) {
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
    // I'm using the 2x2 binned output res of 2028x1520 because it supports 30 fps
    let mut cfg = &mut cfgs.get_mut(0).unwrap();
    cfg.set_size(Size { width: 2028, height: 1520 });
    
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
    while !user_conf.roi_selected {
        // First, check if the user selected the ROI yet (try_recv so we don't block)
        if let Ok((x,y)) = rx_r.try_recv() {
            println!("ROI selection received: x: {x}, y: {y}");
            user_conf.set_roi(x, y);
        }
        // Check the channel for a message, timeout after 2 seconds
        let mut req = rx.recv_timeout(Duration::from_secs(2)).expect("Camera request failed");

        // Get framebuffer for the stream
        let framebuffer: &MemoryMappedFrameBuffer<FrameBuffer> = req.buffer(&stream).unwrap();

        let mut planes: Vec<&[u8]> = framebuffer.data();
        let mut raw_planes: Vec<u8> = Vec::with_capacity(BYTES_PER_RAW_FRAME);
        for p in &planes {
            raw_planes.extend_from_slice(p);
        }
        let cropped_planes: Vec<u8> = crop_frame(planes, y_stride as usize, &user_conf);

        let _ = stream_tx.try_send(cropped_planes);

        req.reuse(ReuseFlag::REUSE_BUFFERS);
        cam.queue_request(req).unwrap();
    }
    (user_conf.crop_x, user_conf.crop_y)
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
    // I'm using the 2x2 binned output res of 2028x1520 because it supports 30 fps
    let mut cfg = &mut cfgs.get_mut(0).unwrap();
    cfg.set_size(Size { width: 2028, height: 1520 });
    
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

    // Set the IMX500 ROI (so that we crop instead of scaling before passing into
    // the NPU).
    match set_roi(&user_conf) {
        Ok(_) => {
            println!(
                "IMX500 ROI set to x:{}, y:{}, w:{}, h:{}\n",
                &user_conf.crop_x,
                &user_conf.crop_y,
                CROP_W, CROP_H,
            );
        },
        Err(_e) => {
            eprintln!("Couldn't set IMX500 ROI!");
            std::process::exit(1);
        }
    }

    // Multiple producer single consumer channel for communication with the recording thread
    let (tx, rx) = channel();

    // Callback executed when frame is captured
    cam.on_request_completed(move |req: Request| {
        if req.status() == RequestStatus::Complete {
            let Some(tensor) = read_imx500_tensor_by_id(&req) else {
                panic!("No tensor len returned, IMX500 isn't working!");
            };
            //println!("{:?}", tensor.len());
        }
        tx.send(req).unwrap();
    });

    // Add all requests to the queue
    for req in reqs {
        cam.queue_request(req).unwrap();
    }

    // Open pipe for ffmpeg conversion
    let (mut reader, mut writer) = pipe().expect("Couldn't open pipe to ffmpeg");

    // Start ffmpeg
    let mut ffmpeg_cmd = Command::new("ffmpeg");
    ffmpeg_cmd
        .args(["-pix_fmt", "yuv420p"])
        .args(["-f", "rawvideo"])
        .args(["-framerate", "30"])
        .args(["-s", format!("{}x{}", CROP_W, CROP_H).as_str()])
        .args(["-i", "pipe:0"])
        .arg(&user_conf.filename)
        .stdin(reader); // pass the read end of the pipe

    let mut ffmpeg_proc = ffmpeg_cmd.spawn().expect("Couldn't start ffmpeg thread");
    // Main loop, loops until user interrupt
    loop {
        // Check the channel for a message, timeout after 2 seconds
        let mut req = rx.recv_timeout(Duration::from_secs(2)).expect("Camera request failed");

        // Get framebuffer for the stream
        let framebuffer: &MemoryMappedFrameBuffer<FrameBuffer> = req.buffer(&stream).unwrap();

        let mut planes: Vec<&[u8]> = framebuffer.data();
        let cropped_planes: Vec<u8> = crop_frame(planes, y_stride as usize, &user_conf);

        writer.write_all(&cropped_planes[..]).expect("Couldn't write frame to ffmpeg pipe");

        req.reuse(ReuseFlag::REUSE_BUFFERS);
        cam.queue_request(req).unwrap();
    }
    drop(writer);
    ffmpeg_proc.wait().expect("Couldn't wrap up ffmpeg");
}

/// Returns a UniquePtr<ControlList> (pretty sure this is an abstraction of a concept from C)
/// with the user-specified configs set
fn global_config(_user_conf: &Config) -> UniquePtr<ControlList> {
    let mut globals = ControlList::new();

    // Unfortunately, with the IMX500, we are clamped to 30 fps :( This may be necessary
    // with a different camera, so I'll leave it here for now.
    // Set framerate (we have to tell it how many microseconds per frame, not a rate in Hz)
    let target_fps = 30.0;
    let frame_duration = (1_000_000.0 / target_fps) as i64;

    globals.set(controls::FrameDurationLimits([frame_duration, frame_duration])).unwrap();

    // Enable caching the input to CNN processing with IMX500 specific control
    // (This just lets us read out CnnInputTensor after the request is complete.)
    globals.set_raw(CNN_ENABLE_INPUT_TENSOR_ID, true.into()).unwrap();
    //println!("{:#?}", globals.get_raw(CNN_ENABLE_INPUT_TENSOR_ID, )); // check if it set

    globals
}

/// Handles frame cropping (for saving or streaming). Needs y_stride from the original Y plane,
/// so that we can account for the zero padding on left/right that the camera does
fn crop_frame<'a>(mut planes: Vec<&'a [u8]>, y_stride: usize, user_conf: &Config) -> Vec<u8> {
    let w: usize;
    let h: usize;
    let x: usize;
    let y: usize;
    if user_conf.roi_selected {
        w = CROP_W as usize;
        h = CROP_H as usize;
        // Make sure the user-defined crop is an even number of pixels in both axes
        x = (user_conf.crop_x & !1) as usize;
        y = (user_conf.crop_y & !1) as usize;
    } else { // if we haven't selected the ROI yet, don't crop as far
        w = FIRST_CROP_W as usize;
        h = FIRST_CROP_H as usize;
        // If we haven't set the ROI yet, default to 480x480 in the center of the frame
        x = 774 as usize;
        y = 520 as usize;
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

/// Read imx500-specific controls by the raw ID (since these are
/// not generated automatically as actual controls in libcamera-rs)
fn read_imx500_tensor_by_id(req: &Request) -> Option<Vec<f32>> {
    let md = req.metadata();
    let md_iter = md.into_iter(); 
    for (id, val) in md_iter {
        match id {
            CNN_OUTPUT_INFO_ID => {
                // Information about the shape and contents of the output tensor, I think
                //println!("{:?}", val);
            },
            CNN_OUTPUT_ID => {
                // The output tensor (which I believe is received as a slice of bytes)
                //println!("Output tensor");
            },
            CNN_INPUT_INFO_ID => {
                // Information about the shape and contents of the input tensor, I think
                //println!("{:?}", val);
            },
            CNN_INPUT_ID => {
                // The input tensor (only works if the enable input tensor control is set)
                // Currently returning this one to the request callback, but that will change
                // to the output tensor once debugging and whatnot is finished.
                if let ControlValue::Byte(bytes) = val {
                    if bytes.len() % 4 == 0 { // make sure value is 32 bit
                        let floats: Vec<f32> =
                            // split the bytes into 4 chunks of 4 and map each to f32
                            bytes.chunks_exact(4) 
                                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                                .collect();
                        return Some(floats);
                    }
                }
            },
            CNN_KPI_INFO_ID => {
                // Information about NPU performance (processing time, etc.)
                //println!("{:#?}", val);
            },
            _ => {
                //println!("Something other than input or output tensors");
            }
        };
    }
    None
}

/// Set custom region of interest for the NPU crop (so we don't just scale the
/// entire frame). This function was drafted by ChatGPT, and I'm no expert on
/// *nix ioctl. However, it seems to work and I think it's ~robust.
///
/// Safety:
///
/// Will crash if the device isn't found or cannot be written to
fn set_roi(user_conf: &Config) -> anyhow::Result<()> {
    // VIDIOC_S/TRY_EXT_CTRLS
    // Uses the macro to define functions in this scope (which we call below)
    // called vidioc_s_ext_ctrls and vidioc_try_ext_ctrls
    nix::ioctl_readwrite!(vidioc_s_ext_ctrls,   b'V', 71, v4l2::v4l2_ext_controls);
    nix::ioctl_readwrite!(vidioc_try_ext_ctrls, b'V', 72, v4l2::v4l2_ext_controls);

    // Open the subdevice as a raw file handle (expected by ioctl)
    let f = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/v4l-subdev2")
        .context("open subdev rw")?;
    let fd = f.as_raw_fd();

    // Per QUERY_EXT_CTRL: elem_size=4, elems=4 -> 16 bytes total
    let mut roi: [u32; 4] = [
        user_conf.crop_x,
        user_conf.crop_y,
        CROP_W, CROP_H,
    ];

    // One control with payload pointer
    // Note: mem::zeroed() initializes a struct with all mem as zeros (unsafe)
    let mut c: v4l2::v4l2_ext_control = unsafe { zeroed() };
    c.id = 9_971_968; // 9971968 "IMX500 Inference Windows"
    c.size = (roi.len() * std::mem::size_of::<u32>()) as u32; // 16

    // These dunder attrs are a mystery to me, currently. Some weird *nix
    // ioctl stuff
    c.__bindgen_anon_1.p_u32 = roi.as_mut_ptr();    // union field

    // Wrapper (note: union for which/ctrl_class; reserved is [u32; 1])
    let mut ctrls: v4l2::v4l2_ext_controls = unsafe { zeroed() };
    ctrls.__bindgen_anon_1.which = v4l2::V4L2_CTRL_WHICH_CUR_VAL; // 0
    ctrls.count = 1;
    ctrls.error_idx = 0;
    ctrls.reserved = [0];     // length 1
    ctrls.request_fd = 0;
    ctrls.controls = &mut c as *mut _;

    unsafe {
        // Optional: validate first
        vidioc_try_ext_ctrls(fd, &mut ctrls).context("VIDIOC_TRY_EXT_CTRLS")?;
        // Apply
        vidioc_s_ext_ctrls(fd, &mut ctrls).context("VIDIOC_S_EXT_CTRLS")?;
    }

    Ok(())
}
