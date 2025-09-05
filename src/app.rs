use crate::{
    // Constants
    RAW_W, RAW_H, BYTES_PER_RAW_FRAME,
    BYTES_PER_RAW_Y_PLANE, BYTES_PER_RAW_UV_PLANE,
};
use crossbeam_channel::{
    Receiver,
};
use eframe::egui::{
    self,
    ColorImage,
    TextureHandle,
    TextureOptions,
};
use std::{
    cmp::max,
    error::Error,
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
    rx: Receiver<Vec<u8>>,
    tex: Option<TextureHandle>,
    last_frame_at: Instant, // Track when the previous frame was captured
}

impl GuiApp {
    pub fn new(ctx: &eframe::CreationContext, rx: Receiver<Vec<u8>>) -> Self {
        Self {
            selection: Selection::default(),
            rx,
            tex: None,
            last_frame_at: Instant::now() 
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
            let size = [2028, 1520];
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
                let tex_size = egui::vec2(RAW_W as f32, RAW_H as f32);
                let scale = (avail.x / tex_size.x).min(avail.y / tex_size.y).max(0.0);
                //let desired = tex_size * scale.max(1.0).min(8.0); // clamp zoom a bit
                ui.image((tex.id(), tex.size_vec2()));
            } else {
                ui.label("Waiting for first frame (RGBA over stdin/FIFO)...");
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
    if num_bytes != BYTES_PER_RAW_FRAME {
        eprintln!("Received the wrong number of bytes while reading the video stream!");
        eprintln!("{num_bytes} != {BYTES_PER_RAW_FRAME}");
        return None
    }

    // Resolution information
    let height: usize = RAW_H as usize;
    let width: usize = RAW_W as usize;
    let y_stride = width + 20; // Raw planes have 10px padding left/right
    let uv_stride = y_stride / 2;


    // Split the planes
    let y_plane: &[u8] = &frame[..BYTES_PER_RAW_Y_PLANE];
    let u_plane: &[u8] = &frame[BYTES_PER_RAW_Y_PLANE..BYTES_PER_RAW_Y_PLANE + BYTES_PER_RAW_UV_PLANE];
    let v_plane: &[u8] = &frame[BYTES_PER_RAW_Y_PLANE + BYTES_PER_RAW_UV_PLANE..];

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
