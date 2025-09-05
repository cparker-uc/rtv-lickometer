use eframe::egui;

// Set font to Hack
pub fn set_font(ctx: &egui::Context) {
    // Load the default fonts
    let mut fonts = egui::FontDefinitions::default();

    // Load Hack from assets
    fonts.font_data.insert(
        "Hack".to_owned(),
        egui::FontData::from_static(include_bytes!(
            "../../assets/Hack-Regular.ttf"
        )),
//      std::sync::Arc::new(egui::FontData::from_static(include_bytes!(
//          "../../assets/Hack-Regular.ttf"
//      ))),
    );

    // Ensure we use Hack for both proportional and monospace
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "Hack".to_owned());
    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .insert(0, "Hack".to_owned());

    // Set the fonts to context
    ctx.set_fonts(fonts);
}

/// Add a bit of padding to the margins (symmetrically)
pub fn set_frame_margins(ctx: &egui::Context) -> egui::Frame {
    let mut frame = egui::Frame::central_panel(&ctx.style());
    frame.inner_margin(egui::Margin::symmetric(16.0,16.0))
}
