fn main() {
    // GUI options
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_resizable(false)
            .with_inner_size([480.0, 580.0])
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
        Box::new(|cc| Ok(Box::new(rtv_lickometer::GuiApp::new(cc)))),
    ).expect("GUI exited unsuccessfully");
}
