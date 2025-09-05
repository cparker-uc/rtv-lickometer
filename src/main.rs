use crossbeam_channel::bounded;
use rtv_lickometer::{
    Config,
    record::{record, stream},
};
use std::{
    sync::mpsc::channel,
    thread,
    time,
};

fn main() {
    // GUI options
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_resizable(false)
            .with_inner_size([1300.0, 800.0])
            .with_icon(
                eframe::icon_data::from_png_bytes(&include_bytes!("../assets/icon_small.png")[..])
                    .expect("Failed to load icon")
            ),
        ..Default::default()
    };

    // Spawn a channel to communicate between recording and rendering threads
    let (tx, rx) = bounded::<Vec<u8>>(2);

    // Start recording in a separate thread
    let first_conf: Config = Config::new(false);
    thread::spawn(move || {
        stream(first_conf, tx);
    });
    // Sleep for a second to let the camera start
    //thread::sleep(time::Duration::from_secs(1));
    
    eframe::run_native(
        "Real-time Video Lickometer",
        native_options,
        Box::new(|cc| Ok(Box::new(rtv_lickometer::GuiApp::new(cc, rx)))),
    ).expect("GUI exited unsuccessfully");
    let second_conf: Config = Config::new(true);
    record(second_conf);
}
