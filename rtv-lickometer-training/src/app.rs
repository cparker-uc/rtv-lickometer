use crate::{
    Config,
    record::{
        gui_stream,
        record,
    },
    RAW_W, RAW_H,
    BYTES_PER_RAW_Y_PLANE,
    BYTES_PER_RAW_UV_PLANE,
    BYTES_PER_RAW_FRAME,
};
use eframe::egui::{
    self,
    ColorImage,
    TextureHandle,
    TextureOptions,
    RichText,
    Color32,
    Vec2,
};
use std::{
    cmp::max,
    sync::{
        mpsc::{
            Sender,
            channel,
        },
    },
    thread::{
        self,
        JoinHandle,
    },
    time::{
        Duration,
        Instant,
    },
};

mod style;

/// This is where we set attributes to persist between loops
///
/// Things are wrapped in Options so that we can take them and
/// drop when wrapping things up. For instance, we take the
/// rx every update, then put it back at the end (unless we
/// want to exit, then we leave it None and update returns
/// immediately each loop).
pub struct GuiApp {
    focus_val: f32,
    rx: Option<crossbeam_channel::Receiver<Vec<u8>>>, // for video frames
    tx_f: Option<Sender<f32>>, // for sending new focus val
    tex: Option<TextureHandle>,
    cam_thread: Option<JoinHandle<()>>, // Ensure that we have wrapped up the camera work before closing
                                // GUI
    filename: Option<String>,
    rec_timer: Option<Instant>, // track when recording started
}

impl GuiApp {
    pub fn new(_ctx: &eframe::CreationContext) -> Self {
        // Spawn a channel to communicate between recording and GUI rendering threads
        let (tx, rx) = crossbeam_channel::bounded::<Vec<u8>>(2);

        // Spawn a channel for setting camera focus
        let (tx_f, rx_f) = channel::<f32>();

        // Start with focus in the middle
        let focus_val = 0.5;

        // Start the camera stream for the GUI
        let cam_thread = thread::spawn(move || {
            // Initialize a new default Config
            let mut user_conf: Config = Config::default();
            user_conf.set_focus(focus_val);

            gui_stream(user_conf, tx, rx_f);
        });

        Self {
            focus_val,
            rx: Some(rx),
            tx_f: Some(tx_f),
            tex: None,
            cam_thread: Some(cam_thread),
            filename: None,
            rec_timer: None,
        }
    }

    /// Until we confirm recording, display a stream of the video
    fn video_player(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(style::set_frame_margins(ctx))
            .show(ctx, |ui| {
            if let Some(tex) = &self.tex {
                // Fit while preserving aspect ratio.
                let curr_frame = ui.image((tex.id(), tex.size_vec2()));

                // Focus slider
                if ui.add(egui::Slider::new(&mut self.focus_val, 0.0..=1.0).text("Camera Focus Slider")).changed() {
                    if let Some(tx_f) = self.tx_f.as_ref() {
                        let _ = tx_f.send(self.focus_val);
                    }
                }

                if ui.button("Start Recording").clicked() {
                    // Wrap up the camera work
                    let cam_thread = self.cam_thread.take().unwrap();
                    cam_thread.join().expect("Couldn't join camera thread");
                    
                    // Shrink the window somewhat, it doesn't need as much real-estate now
                    ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(Vec2::new(480.0, 360.0)));
                }

            }        
        });
    }

    /// Once the confirm button is selected, wrap things up and start recording
    fn confirm_selection_wrapup(&mut self, _ctx: &egui::Context, focus_val: f32) {
        // Once gui_stream exits, set focus to user_conf and start recording.
        let mut user_conf = Config::default();
        user_conf.set_focus(focus_val);

        self.filename = Some(user_conf.filename.to_owned());

        // Spawn a new camera thread and save the JoinHandle
        let camera_thread = thread::spawn(move || {
            record(&user_conf);
        });
        self.cam_thread = Some(camera_thread);

        // Start timing the recording thread so we can report
        // that info to the user
        self.rec_timer = Some(Instant::now());
    }

    /// If we are just recording, on the update loop we want to
    /// display a status message
    fn recording_progress(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(style::set_frame_margins(ctx))
            .show(ctx, |ui| {
                // Filename display
                if let Some(filename) = &self.filename {
                    let filename_label_text = RichText::new("Recording to file: ")
                        .strong()
                        .size(18.0);
                    let filename_text = RichText::new(format!("\t{}\n", filename))
                        //.strong()
                        .italics()
                        .size(16.0);
                    ui.label(filename_label_text);
                    ui.label(filename_text);
                } else {
                    panic!("No filename");
                }
                // Timer display
                if let Some(start_time) = self.rec_timer {
                    let elapsed_time = start_time.elapsed();
                    let min = elapsed_time.as_secs() / 60;  // Floor division
                    let secs = elapsed_time.as_secs() % 60; // don't count minutes
                    let msecs = elapsed_time.subsec_millis();


                    let time_label_text = RichText::new("Time elapsed: ")
                        .strong()
                        .size(18.0);
                    let time_text = RichText::new(format!("\t{}:{}.{}\n", min, secs, msecs))
                        .size(18.0);
                    ui.label(time_label_text);
                    ui.label(time_text);
                    ctx.request_repaint_after(Duration::from_millis(10));
                } else {
                    panic!("Timer didn't start");
                }

                // Stop recording button
                let stop_button_text = RichText::new("Stop Recording")
                    .strong()
                    .color(Color32::RED)
                    .size(18.0);
                let stop_button = ui.button(stop_button_text);

                if stop_button.clicked() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
    }
}

impl eframe::App for GuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // If the camera thread isn't set anymore, wrap up the ROI selection
        // and start the recording thread
        if self.cam_thread.is_none() {
            self.confirm_selection_wrapup(ctx, self.focus_val);
            return;
        }

        // Check if we have received a frame from the camera, otherwise just
        // report the recording progress
        let Some(rx) = self.rx.take() else { 
            self.recording_progress(ctx);
            return; // force return here after reporting progress
        };

        // Pull the newest available frame (drain to most-recent).
        let mut latest: Option<Vec<u8>> = None;

        while let Ok(f) = rx.try_recv() {
            latest = Some(f);
        }
        // If we can convert to RGBA, render the latest frame
        if let Some(rgba) = convert_to_rgba(latest){
            let size = [RAW_W as usize, RAW_H as usize];
            let image = ColorImage::from_rgba_unmultiplied(size, &rgba);
            if let Some(tex) = &mut self.tex {
                tex.set(image, TextureOptions::default());
            } else {
                self.tex = Some(ctx.load_texture("stream", image, TextureOptions::LINEAR));
            }
        } 

        self.video_player(ctx);

        ctx.request_repaint_after(Duration::from_millis(10));

        // return the receiver to its Option
        self.rx = Some(rx);

    }
}

#[inline]
fn clamp8(x: i32) -> u8 {
    if x < 0 { 0 } else if x > 255 { 255 } else { x as u8 }
}

/// Take a raw frame and convert to RGBA for display
fn convert_to_rgba(frame: Option<Vec<u8>>) -> Option<Vec<u8>> {
    frame.as_ref()?;

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
    let y_stride = width;
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
        let cw = width.div_ceil(2); // equiv to (width + 1) / 2

        let u_row = &u_plane[cy * uv_stride .. cy * uv_stride + cw];
        let v_row = &v_plane[cy * uv_stride .. cy * uv_stride + cw];

        for x in 0..width {
            let y_  = y_row[x] as i32;
            let u  = u_row[x / 2] as i32;
            let v  = v_row[x / 2] as i32;

            // Video-range BT.601
            // For full-range, replace with:
            // let c = Y - 0; let d = U - 128; let e = V - 128;
            // let r = (298*c + 409*e + 128) >> 8; etc.
            let c = max(0, y_ - 16);
            let d = u - 128;
            let e = v - 128;

            let r = (298*c + 409*e + 128) >> 8;
            let g = (298*c - 100*d - 208*e + 128) >> 8;
            let b = (298*c + 516*d + 128) >> 8;

            let o = (y * width + x) * 4;
            out[o] = clamp8(r);
            out[o + 1] = clamp8(g);
            out[o + 2] = clamp8(b);
            out[o + 3] = 255;
        }
    }
    Some(out)
}
