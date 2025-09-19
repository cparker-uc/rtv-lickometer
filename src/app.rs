use crate::{
    Config,
    record::{
        load_firmware,
        gui_stream,
        record,
    },
    // Constants
    FIRST_CROP_W, FIRST_CROP_H,
    BYTES_PER_FIRST_CROP_Y_PLANE,
    BYTES_PER_FIRST_CROP_UV_PLANE,
    BYTES_PER_FIRST_CROP_FRAME,
    CROP_W, CROP_H,
};
use eframe::egui::{
    self,
    ColorImage,
    TextureHandle,
    TextureOptions,
    Pos2,
    Rect,
    Vec2,
    Sense,
    RichText,
    Color32,
    Stroke,
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
    selection: (u32, u32),
    rx: Option<crossbeam_channel::Receiver<Vec<u8>>>,
    tx_r: Option<Sender<(u32, u32)>>, // for transmitting selected ROI back
    tex: Option<TextureHandle>,
    //last_frame_at: Instant, // Track when the previous frame was captured
    cam_thread: Option<JoinHandle<()>>, // Ensure that we have wrapped up the camera work before closing
                                // GUI
    filename: Option<String>,
    rec_timer: Option<Instant>, // track when recording started
}

impl GuiApp {
    pub fn new(_ctx: &eframe::CreationContext) -> Self {
        // Spawn a channel to communicate between recording and GUI rendering threads
        let (tx, rx) = crossbeam_channel::bounded::<Vec<u8>>(2);

        // Spawn a channel for the other communication direction (we need to communicate
        // both directions to allow the user to actually set the ROI).
        // Here, the channel will pass a tuple of u32s (x,y).
        let (tx_r, rx_r) = channel::<(u32, u32)>();

        // Start with selection at origin
        let roi_selection = (0u32, 0u32);

        // Start the camera stream for the GUI
        let cam_thread = thread::spawn(move || {
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
            gui_stream(user_conf, tx, rx_r);
        });

        Self {
            selection: roi_selection,
            rx: Some(rx),
            tx_r: Some(tx_r),
            tex: None,
            //last_frame_at: Instant::now(),
            cam_thread: Some(cam_thread),
            filename: None,
            rec_timer: None,
        }
    }

    /// Until we select the ROI, display a stream of the video
    fn video_player(&mut self, ctx: &egui::Context, latest: Vec<u8>) {
        // Create or update texture.
        let size = [FIRST_CROP_W as usize, FIRST_CROP_H as usize];
        let image = ColorImage::from_rgba_unmultiplied(size, &latest);
        if let Some(tex) = &mut self.tex {
            tex.set(image, TextureOptions::default());
        } else {
            self.tex = Some(ctx.load_texture("stream", image, egui::TextureOptions::LINEAR));
        }
        //self.last_frame_at = Instant::now();
        ctx.request_repaint();

        egui::CentralPanel::default()
            .frame(style::set_frame_margins(ctx))
            .show(ctx, |ui| {
            if let Some(tex) = &self.tex {
                // Fit while preserving aspect ratio.
                let curr_frame = ui.image((tex.id(), tex.size_vec2()));

                let (x, y) = self.selection;
                // Draw the ROI selection rectangle. Start by determining bounds on the
                // current frame. We will need to subtract the top left point of the
                // frame from the selection top left point (because there is a bit of
                // padding around the frame in the GUI).
                let frame_rect = curr_frame.rect;
                let frame_top_left = frame_rect.min;
                let mut top_left = Pos2::new(x as f32, y as f32);
                let mut roi_rect = Rect::from_min_size(top_left, Vec2::new(CROP_W as f32, CROP_H as f32));

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
                    self.selection = (top_left.x as u32, top_left.y as u32);
                }

                // Clamp the ROI rectangle to the frame
                let min = frame_rect.min;
                let max = frame_rect.max - Vec2::new(CROP_W as f32, CROP_H as f32);

                top_left.x = top_left.x.clamp(min.x, max.x.max(min.x));
                top_left.y = top_left.y.clamp(min.y, max.y.max(min.y));
                roi_rect = Rect::from_min_size(top_left, Vec2::new(CROP_W as f32, CROP_H as f32));

                // Paint the rectangle
                let p = ui.painter();
                p.rect_stroke(roi_rect, 3.0, Stroke::new(4.0, Color32::RED));

                // On confirm button click, we compute the top left of the user-selected ROI
                // (this involves subtracting off the top left of the frame to handle padding
                // in the GUI). Then we send this over the tx_r channel to the camera thread.
                if ui.button("Confirm ROI").clicked() {
                    let top_left = top_left - frame_top_left;
                    let top_left: (u32, u32) = (top_left.x as u32, top_left.y as u32);
                    let tx_r = self.tx_r.take().unwrap();
                    let _ = tx_r.send(top_left); // ignore error if we can't send (keep trying)

                    // Wrap up the camera work
                    let cam_thread = self.cam_thread.take().unwrap();
                    cam_thread.join().expect("Couldn't join camera thread");
                    
                    // Drop the channels to ensure we wrap up nicely.
                    drop(tx_r);

                    // Shrink the window somewhat, it doesn't need as much real-estate now
                    ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(Vec2::new(480.0, 360.0)));
                }

            }        
        });
    }

    /// Once the ROI is selected, wrap things up and start recording
    fn roi_selection_wrapup(&mut self, _ctx: &egui::Context) {
        // Once gui_stream exits, set the ROI that was returned
        // and start recording.
        let mut user_conf = Config::default();
        let (x,y) = self.selection;
        user_conf.set_roi(x, y);

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
                // Report the selected ROI
                ui.label(RichText::new("ROI selected: ").strong().size(18.0));
                ui.label(
                    RichText::new(
                        format!("\tx: {}, y: {}, w: {}, h: {}\n", self.selection.0, self.selection.1, 224, 224)
                    ).size(16.0)
                );

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
            self.roi_selection_wrapup(ctx);
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
        if let Some(latest) = convert_to_rgba(latest){
            self.video_player(ctx, latest);
            // Check tx_r now, and return early to skip replacing rx
            // if tx_r is gone
            if self.tx_r.is_none() { return; }
        } else {
            // No frame this tick; to avoid busy-looping, request a repaint soon.
            ctx.request_repaint_after(Duration::from_millis(10));
        }

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
    //if frame.is_none() { return None }
    frame.as_ref()?;

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
