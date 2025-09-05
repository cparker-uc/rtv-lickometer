// Module definitions and re-exports
mod app;
pub use app::GuiApp;
pub mod record;

use hostname; // For setting output filename based on hostname
use chrono;

// These should be constants because the camera can only record
// at 30 fps in one resolution, and the NN will always need the same shape
pub const RAW_W: u32 = 2028;
pub const RAW_H: u32 = 1520;
// Raw planes will have 10px padding on left/right
pub const BYTES_PER_RAW_Y_PLANE: usize = ((RAW_W + 20) * RAW_H) as usize;
pub const BYTES_PER_RAW_UV_PLANE: usize = ((RAW_W + 20) / 2 * RAW_H / 2) as usize;
pub const BYTES_PER_RAW_FRAME: usize = BYTES_PER_RAW_Y_PLANE + 2 * BYTES_PER_RAW_UV_PLANE;

// Intermediate crop size (for allowing the user to select the crop rectangle)
pub const FIRST_CROP_W: u32 = 480;
pub const FIRST_CROP_H: u32 = 480;
pub const BYTES_PER_FIRST_CROP_Y_PLANE: usize = (FIRST_CROP_W * FIRST_CROP_H) as usize;
pub const BYTES_PER_FIRST_CROP_UV_PLANE: usize = (FIRST_CROP_W / 2 * FIRST_CROP_H / 2) as usize;
pub const BYTES_PER_FIRST_CROP_FRAME: usize = BYTES_PER_FIRST_CROP_Y_PLANE + 2 * BYTES_PER_FIRST_CROP_UV_PLANE;

// Final crop size (currently 100 px padded width/height so we can jitter the ROI in network
// training)
pub const CROP_W: u32 = 324;
pub const CROP_H: u32 = 324;
pub const BYTES_PER_CROPPED_Y_PLANE: usize = (CROP_W * CROP_H) as usize;
pub const BYTES_PER_CROPPED_UV_PLANE: usize = (CROP_W / 2 * CROP_H / 2) as usize;
pub const BYTES_PER_CROPPED_FRAME: usize = BYTES_PER_CROPPED_Y_PLANE + 2 * BYTES_PER_CROPPED_UV_PLANE;


/// Contains various user-exposed configuration options
///
/// Currently only filename and crop rectangle, likely will add more later
pub struct Config {
    filename: String,
    //cnn_path: 
    crop_x: u32,
    crop_y: u32,
    roi_selected: bool, // flag to track if we are in selection GUI
}

impl Config {
    /// Initialize the configuration with a descriptive filename and default crop
    pub fn new(roi_selected: bool) -> Self {
        // Default: Crop from 2028x1520 to 480x480 (centered in the frame)
        // This allows a bit of room for the user to configure the ROI
        let crop_x: u32 = 774;
        let crop_y: u32 = 520;

        // Filename is required, starts empty
        let hostname = hostname::get()
            .unwrap()
            .into_string()
            .unwrap_or_else(|_| "unknown-hostname".to_string());
        let date = chrono::Local::now().format("%Y-%m-%d").to_string();
        let filename = format!("{}_{}.mp4", hostname, date);

        Self { filename, crop_x, crop_y, roi_selected }
    }
}

/// Prints the usage information when the user passes invalid args or the help flag
fn print_help() {
    println!("Usage: ./imx500-test [OPTIONS] </path/to/output>");
    println!(
        "Available options:

-h/--help   :   Print this help message
-c/--crop   :   Specify the crop rectangle as X,Y,W,H

Examples:

./imx500-test --help
./imx500-test --crop=0,0,1920,1080 output.raw"
                    );
}
