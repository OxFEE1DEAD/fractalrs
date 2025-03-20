use eframe::egui;
use egui::{ViewportBuilder, Vec2, Pos2};
use image::{ImageBuffer, Rgb};
use num_complex::Complex64;
use rayon::prelude::*;
use parking_lot::RwLock;
use std::sync::Arc;
use chrono::Local;
use num_cpus;
use rand::Rng;

#[derive(Clone, Copy, PartialEq)]
enum FractalType {
    Classic,
    Spiral,
    Flower,
    Phoenix,
    Butterfly,
}

struct FractalState {
    fractal_type: FractalType,
    zoom: f64,
    center_x: f64,
    center_y: f64,
    max_iter: u32,
    hue_offset: f32,
    saturation: f32,
    value: f32,
    width: u32,
    height: u32,
    needs_update: bool,
    power: f64,
    secondary_param: f64,  // For additional variations
}

struct FractalApp {
    state: Arc<RwLock<FractalState>>,
    image_texture: Option<egui::TextureHandle>,
    drag_start: Option<Pos2>,
    drag_start_center: Option<(f64, f64)>,
    thread_count: usize,
}

impl Default for FractalApp {
    fn default() -> Self {
        Self {
            state: Arc::new(RwLock::new(FractalState {
                fractal_type: FractalType::Classic,
                zoom: 1.0,
                center_x: -0.5,
                center_y: 0.0,
                max_iter: 1000,
                hue_offset: 0.0,
                saturation: 1.0,
                value: 1.0,
                width: 800,
                height: 600,
                needs_update: true,
                power: 2.0,
                secondary_param: 0.5,
            })),
            image_texture: None,
            drag_start: None,
            drag_start_center: None,
            thread_count: num_cpus::get(),
        }
    }
}

impl FractalApp {
    fn iterate_fractal(&self, c: Complex64, state: &FractalState) -> u32 {
        let mut z = Complex64::new(0.0, 0.0);
        let power = state.power;
        let param = state.secondary_param;

        match state.fractal_type {
            FractalType::Classic => {
                for i in 0..state.max_iter {
                    if z.norm_sqr() > 4.0 {
                        return i;
                    }
                    z = z.powf(power) + c;
                }
            }
            FractalType::Spiral => {
                let mut prev = z;
                for i in 0..state.max_iter {
                    if z.norm_sqr() > 4.0 {
                        return i;
                    }
                    let temp = z;
                    z = z.powf(power) + c + (prev * param);
                    prev = temp;
                }
            }
            FractalType::Flower => {
                for i in 0..state.max_iter {
                    if z.norm_sqr() > 4.0 {
                        return i;
                    }
                    z = (z * z.sin() + c) * Complex64::new(param.cos(), param.sin());
                }
            }
            FractalType::Phoenix => {
                let mut prev = z;
                for i in 0..state.max_iter {
                    if z.norm_sqr() > 4.0 {
                        return i;
                    }
                    let temp = z;
                    z = z.powf(power) - prev.sin() * param + c;
                    prev = temp;
                }
            }
            FractalType::Butterfly => {
                for i in 0..state.max_iter {
                    if z.norm_sqr() > 4.0 {
                        return i;
                    }
                    let r = z.norm();
                    if r > 0.0 {
                        let theta = z.arg();
                        z = Complex64::from_polar(r.powf(param), theta * power) + c;
                    }
                }
            }
        }
        state.max_iter
    }

    fn generate_mandelbrot(&self) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let state = self.state.read();
        let mut img = ImageBuffer::new(state.width, state.height);
        let scale = 2.5 / state.zoom;
        
        let chunks: Vec<_> = (0..state.height)
            .collect::<Vec<_>>()
            .chunks(state.height as usize / self.thread_count + 1)
            .map(|c| c.to_vec())
            .collect();

        let results: Vec<_> = chunks.into_par_iter().map(|rows| {
            let mut buffer = Vec::new();
            for y in rows {
                for x in 0..state.width {
                    let x_scaled = (x as f64 / state.width as f64) * 3.5 * scale - 2.5 * scale + state.center_x;
                    let y_scaled = (y as f64 / state.height as f64) * 2.0 * scale - 1.0 * scale + state.center_y;
                    
                    let c = Complex64::new(x_scaled, y_scaled);
                    let i = self.iterate_fractal(c, &state);
                    
                    let hue = ((i as f32 / state.max_iter as f32) * 360.0 + state.hue_offset) % 360.0;
                    let color = if i == state.max_iter {
                        Rgb([0, 0, 0])
                    } else {
                        let rgb = self.hsv_to_rgb(hue, state.saturation, state.value);
                        Rgb([rgb.0, rgb.1, rgb.2])
                    };
                    buffer.push((x, y, color));
                }
            }
            buffer
        }).collect();

        for chunk in results {
            for (x, y, color) in chunk {
                img.put_pixel(x, y, color);
            }
        }
        
        img
    }

    #[inline(always)]
    fn mandelbrot(&self, c: Complex64, max_iter: u32) -> u32 {
        let mut z = Complex64::new(0.0, 0.0);
        for i in 0..max_iter {
            if z.norm_sqr() > 4.0 {
                return i;
            }
            z = z * z + c;
        }
        max_iter
    }

    #[inline(always)]
    fn hsv_to_rgb(&self, h: f32, s: f32, v: f32) -> (u8, u8, u8) {
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;
        
        let (r, g, b) = match h as i32 {
            h if h < 60 => (c, x, 0.0),
            h if h < 120 => (x, c, 0.0),
            h if h < 180 => (0.0, c, x),
            h if h < 240 => (0.0, x, c),
            h if h < 300 => (x, 0.0, c),
            _ => (c, 0.0, x)
        };
        
        (((r + m) * 255.0) as u8,
         ((g + m) * 255.0) as u8,
         ((b + m) * 255.0) as u8)
    }

    fn handle_mouse_input(&mut self, ui: &mut egui::Ui, available_size: Vec2) {
        let rect = ui.max_rect();
        let response = ui.allocate_rect(rect, egui::Sense::drag());
        
        if response.dragged() {
            if let Some(drag_start) = self.drag_start {
                if let Some((start_x, start_y)) = self.drag_start_center {
                    let delta = response.drag_delta();
                    let mut state = self.state.write();
                    let scale = 2.5 / state.zoom;
                    let sensitivity = 0.5;
                    let dx = (delta.x as f64) * scale * sensitivity / (available_size.x as f64);
                    let dy = (delta.y as f64) * scale * sensitivity / (available_size.y as f64);
                    state.center_x = start_x - dx;
                    state.center_y = start_y - dy;
                    state.needs_update = true;
                }
            } else {
                let state = self.state.read();
                self.drag_start = Some(response.interact_pointer_pos().unwrap());
                self.drag_start_center = Some((state.center_x, state.center_y));
            }
        } else {
            self.drag_start = None;
            self.drag_start_center = None;
        }

        if response.hovered() {
            ui.input(|i| {
                let scroll = i.raw_scroll_delta.y;
                if scroll != 0.0 {
                    let mut state = self.state.write();
                    let zoom_factor = if scroll > 0.0 { 1.05 } else { 0.95 };
                    let new_zoom = state.zoom * zoom_factor;
                    
                    if new_zoom >= 0.1 && new_zoom <= 50.0 {
                        state.zoom = new_zoom;
                        state.needs_update = true;
                    }
                }
            });
        }
    }

    fn randomize_params(&mut self) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let mut state = self.state.write();
        state.hue_offset = rng.gen_range(0.0..360.0);
        state.saturation = rng.gen_range(0.7..1.0);
        state.value = rng.gen_range(0.7..1.0);
        state.power = rng.gen_range(2.0..4.0);
        state.secondary_param = rng.gen_range(0.1..0.9);
        state.fractal_type = match rng.gen_range(0..5) {
            0 => FractalType::Classic,
            1 => FractalType::Spiral,
            2 => FractalType::Flower,
            3 => FractalType::Phoenix,
            _ => FractalType::Butterfly,
        };
        state.needs_update = true;
    }
}

impl eframe::App for FractalApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("controls").show(ctx, |ui| {
            ui.heading("Fractal Controls");
            
            let mut state = self.state.write();
            
            ui.horizontal(|ui| {
                ui.label("Fractal Type:");
                if ui.radio_value(&mut state.fractal_type, FractalType::Classic, "Classic").clicked() {
                    state.needs_update = true;
                }
                if ui.radio_value(&mut state.fractal_type, FractalType::Spiral, "Spiral").clicked() {
                    state.needs_update = true;
                }
                if ui.radio_value(&mut state.fractal_type, FractalType::Flower, "Flower").clicked() {
                    state.needs_update = true;
                }
                if ui.radio_value(&mut state.fractal_type, FractalType::Phoenix, "Phoenix").clicked() {
                    state.needs_update = true;
                }
                if ui.radio_value(&mut state.fractal_type, FractalType::Butterfly, "Butterfly").clicked() {
                    state.needs_update = true;
                }
            });

            ui.add_space(10.0);
            
            if ui.button("🎲 Randomize").clicked() {
                drop(state);  // Release the lock before calling randomize
                self.randomize_params();
                state = self.state.write();  // Reacquire the lock
            }

            ui.add_space(5.0);
            
            if ui.add(egui::Slider::new(&mut state.power, 2.0..=4.0)
                .step_by(0.1)
                .text("Power")).changed() {
                state.needs_update = true;
            }
            
            if ui.add(egui::Slider::new(&mut state.secondary_param, 0.1..=0.9)
                .step_by(0.05)
                .text("Shape Parameter")).changed() {
                state.needs_update = true;
            }
            if ui.add(egui::Slider::new(&mut state.zoom, 0.1..=50.0)
                .step_by(0.1)
                .text("Zoom")).changed() {
                state.needs_update = true;
            }
            if ui.add(egui::Slider::new(&mut state.center_x, -2.0..=1.0)
                .step_by(0.01)
                .text("X Position")).changed() {
                state.needs_update = true;
            }
            if ui.add(egui::Slider::new(&mut state.center_y, -1.5..=1.5)
                .step_by(0.01)
                .text("Y Position")).changed() {
                state.needs_update = true;
            }
            if ui.add(egui::Slider::new(&mut state.max_iter, 100..=5000)
                .step_by(100.0)
                .text("Max Iterations")).changed() {
                state.needs_update = true;
            }
            
            ui.separator();
            ui.heading("Color Controls");
            if ui.add(egui::Slider::new(&mut state.hue_offset, 0.0..=360.0).text("Hue Offset")).changed() {
                state.needs_update = true;
            }
            if ui.add(egui::Slider::new(&mut state.saturation, 0.0..=1.0).text("Saturation")).changed() {
                state.needs_update = true;
            }
            if ui.add(egui::Slider::new(&mut state.value, 0.0..=1.0).text("Value")).changed() {
                state.needs_update = true;
            }
            
            if ui.button("Save Image").clicked() {
                let img = self.generate_mandelbrot();
                let filename = format!("fractol_{}.png", 
                    Local::now().format("%Y%m%d_%H%M%S"));
                img.save(&filename).unwrap();
            }
            
            ui.separator();
            ui.heading("Controls");
            ui.label("• Drag to pan");
            ui.label("• Scroll to zoom");
            ui.label("• Use sliders for fine control");
            ui.label(format!("Using {} threads", self.thread_count));
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let available_size = ui.available_size();
            let needs_update = {
                let mut state = self.state.write();
                let size_changed = state.width != available_size.x as u32 || 
                                 state.height != available_size.y as u32;
                
                if size_changed {
                    state.width = available_size.x as u32;
                    state.height = available_size.y as u32;
                    state.needs_update = true;
                }
                
                let needs_update = state.needs_update;
                state.needs_update = false;
                needs_update
            };

            if needs_update {
                let img = self.generate_mandelbrot();
                let color_image = egui::ColorImage::from_rgb(
                    [self.state.read().width as usize, self.state.read().height as usize],
                    img.as_raw()
                );
                
                let texture = self.image_texture.get_or_insert_with(|| {
                    ui.ctx().load_texture(
                        "mandelbrot",
                        color_image.clone(),
                        Default::default()
                    )
                });
                
                texture.set(color_image, Default::default());
            }
            
            if let Some(texture) = &self.image_texture {
                ui.add(egui::Image::new(&*texture).fit_to_original_size(1.0));
            }
            
            self.handle_mouse_input(ui, available_size);
        });
        
        // Request continuous repaint only when dragging or recent updates
        if self.drag_start.is_some() {
            ctx.request_repaint();
        }
    }
}

fn main() {
    let options = eframe::NativeOptions {
        viewport: ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };
    
    eframe::run_native(
        "Fractal Explorer",
        options,
        Box::new(|_cc| Box::new(FractalApp::default())),
    ).unwrap();
}
