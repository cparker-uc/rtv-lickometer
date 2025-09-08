use crossbeam_channel::bounded;
use rtv_lickometer::{
    Config,
    record::{
        record,
        gui_stream,
        load_firmware,
    },
};
use std::{
    sync::mpsc::channel,
    thread,
    time,
};

fn main() {
    // Initialize a new default Config
    let mut user_conf: Config = Config::default();

    // Spawn a channel to communicate between recording and GUI rendering threads
    let (tx, rx) = bounded::<Vec<u8>>(2);

    // Spawn a channel for the other communication direction (we need to communicate
    // both directions to allow the user to actually set the ROI).
    // Here, the channel will pass a tuple of u32s (x,y).
    let (tx_r, rx_r) = channel::<(u32, u32)>();

    // Start recording in a separate thread. This will contain all the
    // configuration and whatnot, so that we don't have to reinit a
    // CameraManager (which libcamera doesn't like).
    let camera_thread = thread::spawn(move || {
        // Load firmware with V4L2 into IMX500 (the CNN rpk file)
        match load_firmware(&user_conf) {
            Ok(_) => println!("IMX500 finished loading CNN"),
            Err(_e) => {
                eprintln!("Couldn't load CNN into IMX500!");
                std::process::exit(1);
            },
        }
        let (roi_x, roi_y) = gui_stream(&mut user_conf, tx, rx_r);

        // Once gui_stream exits, set the ROI that was returned
        // and start recording.
        record(&user_conf);
    });

    // GUI options
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_resizable(false)
            .with_inner_size([800.0, 600.0])
            .with_icon(
                eframe::icon_data::from_png_bytes(&include_bytes!("../assets/icon_small.png")[..])
                    .expect("Failed to load icon")
            ),
        ..Default::default()
    };

    eframe::run_native(
        "Real-time Video Lickometer",
        native_options,
        Box::new(|cc| Ok(Box::new(rtv_lickometer::GuiApp::new(cc, rx, tx_r)))),
    ).expect("GUI exited unsuccessfully");

}
