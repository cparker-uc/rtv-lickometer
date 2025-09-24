use eframe::egui;

/// Get rid of any padding
pub fn set_frame_margins(ctx: &egui::Context) -> egui::Frame {
    let frame = egui::Frame::central_panel(&ctx.style());
    frame.inner_margin(egui::Margin::symmetric(0.0,0.0))
}
