use crate::{
    // Constants
    //RAW_W, RAW_H, BYTES_PER_RAW_FRAME,
    //BYTES_PER_RAW_Y_PLANE, BYTES_PER_RAW_UV_PLANE,
    FIRST_CROP_W, FIRST_CROP_H,
    BYTES_PER_FIRST_CROP_Y_PLANE,
    BYTES_PER_FIRST_CROP_UV_PLANE,
    BYTES_PER_FIRST_CROP_FRAME,
    CROP_W, CROP_H,
};
use crossbeam_channel;
use eframe::egui::{
    self,
    ColorImage,
    TextureHandle,
    TextureOptions,
    Pos2,
    Rect,
    Vec2,
    Sense,
};
use std::{
    cmp::max,
    error::Error,
    sync::mpsc::Sender,
    time::{
        Duration,
        Instant,
    },
};

mod style;

/// Hold the coordinates of the user-defined selection
pub struct Selection {
    x: usize,
    y: usize,
}

impl Selection {
    /// Default constructor
    pub fn default() -> Self {
        // Default selection values
        let x: usize = 0;
        let y: usize = 0;

        Self { x, y }
    }

    // Getter and setter
    pub fn get_selection(&self) -> [usize; 2] {
        [self.x, self.y]
    }

    /// selection should be a slice of 2 usize in the order x, y
    pub fn set_selection(&mut self, selection: &[usize]) -> Result<(), Box<dyn Error>> {
        self.x = selection[0];
        self.y = selection[1];
        Ok(())
    }

}

/// This is where we set attributes to persist between loops
pub struct GuiApp {
    pub selection: Selection,
    rx: crossbeam_channel::Receiver<Vec<u8>>,
    tx_r: Sender<(u32, u32)>, // for transmitting selected ROI back
    tex: Option<TextureHandle>,
    last_frame_at: Instant, // Track when the previous frame was captured
}

impl GuiApp {
    pub fn new(ctx: &eframe::CreationContext, rx: crossbeam_channel::Receiver<Vec<u8>>, tx_r: Sender<(u32, u32)>) -> Self {
        Self {
            selection: Selection::default(),
            rx,
            tx_r,
            tex: None,
            last_frame_at: Instant::now(),
        }
    }
}

impl eframe::App for GuiApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // Pull the newest available frame (drain to most-recent).
        let mut latest: Option<Vec<u8>> = None;
        while let Ok(f) = self.rx.try_recv() {
            latest = Some(f);
        }
        let latest = convert_to_rgba(latest);

        if let Some(pixels) = latest {
            // Create or update texture.
            let size = [FIRST_CROP_W as usize, FIRST_CROP_H as usize];
            let image = ColorImage::from_rgba_unmultiplied(size, &pixels);
            if let Some(tex) = &mut self.tex {
                tex.set(image, TextureOptions::default());
            } else {
                self.tex = Some(ctx.load_texture("stream", image, egui::TextureOptions::LINEAR));
            }
            self.last_frame_at = Instant::now();
            ctx.request_repaint(); // keep pumping
        } else {
            // No frame this tick; to avoid busy-looping, request a repaint soon.
            ctx.request_repaint_after(Duration::from_millis(10));
        }

        egui::CentralPanel::default()
            .frame(style::set_frame_margins(ctx))
            .show(ctx, |ui| {
            if let Some(tex) = &self.tex {
                // Fit while preserving aspect ratio.
                let avail = ui.available_size();
                // need to subtract the padding bytes from the width
                let tex_size = egui::vec2(FIRST_CROP_W as f32, FIRST_CROP_H as f32);
                //let scale = (avail.x / tex_size.x).min(avail.y / tex_size.y).max(0.0);
                //let desired = tex_size * scale.max(1.0).min(8.0); // clamp zoom a bit
                let curr_frame = ui.image((tex.id(), tex.size_vec2()));

                // Draw the ROI selection rectangle. Start by determining bounds on the
                // current frame. We will need to subtract the top left point of the
                // frame from the selection top left point (because there is a bit of
                // padding around the frame in the GUI).
                let frame_rect = curr_frame.rect;
                let frame_top_left = frame_rect.min;
                ui.label(format!("{frame_top_left:#?}")); // DEBUG print
                let mut top_left = Pos2::new(self.selection.x as f32, self.selection.y as f32);
                ui.label(format!("{top_left:#?}")); // DEBUG print
                let mut roi_rect = Rect::from_min_size(top_left, Vec2::new(CROP_W as f32, CROP_H as f32));
                ui.label(format!("{roi_rect:#?}")); // DEBUG print

                // Get a persistent ID for the ROI rectangle
                let id = ui.make_persistent_id("roi-rect");
                
                // Define a draggable widget with the roi_rect shape and id
                let resp = ui.interact(roi_rect, id, Sense::click_and_drag());

                // On interaction
                if resp.dragged() {
                    // How far did the pointer move during the drag?
                    let delta = ui.input(|i| i.pointer.delta());
                    // Change the top-left coord based on the delta
                    top_left += delta;
                }

                let confirm_button = ui.button("Confirm ROI");

                // On confirm button click, we compute the top left of the user-selected ROI
                // (this involves subtracting off the top left of the frame to handle padding
                // in the GUI). Then we send this over the tx_r channel to the camera thread.
                if confirm_button.clicked() {
                    let top_left = top_left - frame_top_left;
                    let top_left: (u32, u32) = (top_left.x as u32, top_left.y as u32);
                    self.tx_r.send(top_left);
                }

            } else {
                ui.label("Waiting for first frame...");
            }


        });
    }
}

#[inline]
fn clamp8(x: i32) -> u8 {
    if x < 0 { 0 } else if x > 255 { 255 } else { x as u8 }
}

/// Take a raw frame and convert to RGBA for display
fn convert_to_rgba(frame: Option<Vec<u8>>) -> Option<Vec<u8>> {
    if frame.is_none() { return None }

    let frame = frame.unwrap();
    let num_bytes = frame.len();
    if num_bytes != BYTES_PER_FIRST_CROP_FRAME {
        eprintln!("Received the wrong number of bytes while reading the video stream!");
        eprintln!("{num_bytes} != {BYTES_PER_FIRST_CROP_FRAME}");
        return None
    }

    // Resolution information
    let height: usize = FIRST_CROP_H as usize;
    let width: usize = FIRST_CROP_W as usize;
    let y_stride = width;
    let uv_stride = y_stride / 2;


    // Split the planes
    let y_plane: &[u8] = &frame[..BYTES_PER_FIRST_CROP_Y_PLANE];
    let u_plane: &[u8] = &frame[BYTES_PER_FIRST_CROP_Y_PLANE..BYTES_PER_FIRST_CROP_Y_PLANE + BYTES_PER_FIRST_CROP_UV_PLANE];
    let v_plane: &[u8] = &frame[BYTES_PER_FIRST_CROP_Y_PLANE + BYTES_PER_FIRST_CROP_UV_PLANE..];

    // We need 4 bytes per pixel to do RGBA
    let mut out = vec![0u8; height * width * 4];

    for y in 0..height {
        let y_row = &y_plane[y * y_stride .. y * y_stride + width];

        // chroma coordinates (subsampled 2x2)
        let cy = y / 2;
        let cw = (width + 1) / 2;

        let u_row = &u_plane[cy * uv_stride .. cy * uv_stride + cw];
        let v_row = &v_plane[cy * uv_stride .. cy * uv_stride + cw];

        for x in 0..width {
            let Y  = y_row[x] as i32;
            let U  = u_row[x / 2] as i32;
            let V  = v_row[x / 2] as i32;

            // Video-range BT.601
            // For full-range, replace with:
            // let c = Y - 0; let d = U - 128; let e = V - 128;
            // let r = (298*c + 409*e + 128) >> 8; etc.
            let c = max(0, Y - 16);
            let d = U - 128;
            let e = V - 128;

            let r = (298*c + 409*e + 128) >> 8;
            let g = (298*c - 100*d - 208*e + 128) >> 8;
            let b = (298*c + 516*d + 128) >> 8;

            let o = (y * width + x) * 4;
            out[o + 0] = clamp8(r);
            out[o + 1] = clamp8(g);
            out[o + 2] = clamp8(b);
            out[o + 3] = 255;
        }
    }
    Some(out)
}
