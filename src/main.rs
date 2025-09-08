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
    sync::{
        Arc,
        mpsc::channel,
        Mutex,
    },
    thread,
    time,
};

fn main() {
    // Spawn a channel to communicate between recording and GUI rendering threads
    let (tx, rx) = bounded::<Vec<u8>>(2);

    // Spawn a channel for the other communication direction (we need to communicate
    // both directions to allow the user to actually set the ROI).
    // Here, the channel will pass a tuple of u32s (x,y).
    let (tx_r, rx_r) = channel::<(u32, u32)>();

    // Mutex for selection coords
    let roi_selection = Arc::new(Mutex::new((0u32, 0u32)));
    // Cloned Mutex to pass for writing ROI
    let roi_writer = Arc::clone(&roi_selection);

    // Start the camera stream for the GUI
    let camera_thread = thread::spawn(move || {
        // Initialize a new default Config
        let user_conf: Config = Config::default();

        // Load firmware with V4L2 into IMX500 (the CNN rpk file)
        match load_firmware(&user_conf) {
            Ok(_) => println!("IMX500 finished loading CNN"),
            Err(_e) => {
                eprintln!("Couldn't load CNN into IMX500!");
                std::process::exit(1);
            },
        }
        let (x, y) = gui_stream(user_conf, tx, rx_r);
        let mut roi = roi_writer.lock().unwrap();
        roi.0 = x;
        roi.1 = y;
        println!("End of cam_thread");
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
        run_and_return: true,
        ..Default::default()
    };

    eframe::run_native(
        "Real-time Video Lickometer",
        native_options,
        Box::new(|cc| Ok(Box::new(rtv_lickometer::GuiApp::new(cc, rx, tx_r, camera_thread)))),
    ).expect("GUI exited unsuccessfully");

    // Once gui_stream exits, set the ROI that was returned
    // and start recording.
    let mut user_conf = Config::default();
    let roi_data = roi_selection.lock().unwrap();
    let roi_x = roi_data.0;
    let roi_y = roi_data.1;
    user_conf.set_roi(roi_x, roi_y);

    let camera_thread = thread::spawn(move || {
        record(&user_conf);
    });
    camera_thread.join().expect("Couldn't join camera thread.");
}
