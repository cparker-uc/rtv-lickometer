// Module definitions and re-exports
mod app;
pub use app::GuiApp;
pub mod record;

pub const RAW_W: u32 = 1536;
pub const RAW_H: u32 = 864;
// Raw planes will have 10px padding on left/right
pub const BYTES_PER_RAW_Y_PLANE: usize = ((RAW_W + 20) * RAW_H) as usize;
pub const BYTES_PER_RAW_UV_PLANE: usize = ((RAW_W + 20) / 2 * RAW_H / 2) as usize;
pub const BYTES_PER_RAW_FRAME: usize = BYTES_PER_RAW_Y_PLANE + 2 * BYTES_PER_RAW_UV_PLANE;

/// Contains various user-exposed configuration options
///
/// Currently only filename and camera focus, likely will add more later
pub struct Config {
    filename: String,
    focus: f32,
}

impl Config {
    pub fn set_focus(&mut self, focus_val: f32) {
        self.focus = focus_val;
    }
}

impl Default for Config {
    /// Initialize the configuration with a descriptive filename and default crop
    fn default() -> Self {
        let focus: f32 = 0.5;

        // Filename is required, starts empty
        let hostname = hostname::get()
            .unwrap()
            .into_string()
            .unwrap_or_else(|_| "unknown-hostname".to_string());
        let date = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
        let filename = format!("{}_{}.mp4", hostname, date);

        Self { filename, focus }
    }

}
