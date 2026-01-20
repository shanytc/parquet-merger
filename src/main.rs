use eframe::egui;
use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use walkdir::WalkDir;

use arrow::array::{Array, RecordBatch};
use arrow::datatypes::{DataType, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_min_inner_size([600.0, 400.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Parquet Merger",
        options,
        Box::new(|_cc| Ok(Box::new(ParquetMergerApp::default()))),
    )
}

/// Represents a parquet file with both full path and display path
#[derive(Clone)]
struct ParquetFile {
    /// Full absolute path to the file
    full_path: PathBuf,
    /// Relative path for display (e.g., "subfolder/file.parquet")
    display_path: String,
}

#[derive(Default)]
struct ParquetMergerApp {
    /// List of folders to scan
    folders: Vec<PathBuf>,
    /// Found parquet files after scanning
    parquet_files: Vec<ParquetFile>,
    /// Set of selected file indices for merging
    selected_files: HashSet<usize>,
    /// Status message to display
    status_message: String,
    /// Whether an operation is in progress
    is_processing: bool,
    /// Whether to also export to CSV
    export_to_csv: bool,
}

impl ParquetMergerApp {
    fn add_folder(&mut self) {
        if let Some(folder) = rfd::FileDialog::new().pick_folder() {
            if !self.folders.contains(&folder) {
                self.folders.push(folder);
                self.status_message = "Folder added. Click 'Scan' to find parquet files.".to_string();
            } else {
                self.status_message = "Folder already in list.".to_string();
            }
        }
    }

    fn remove_folder(&mut self, index: usize) {
        if index < self.folders.len() {
            self.folders.remove(index);
            self.parquet_files.clear();
            self.selected_files.clear();
            self.status_message = "Folder removed.".to_string();
        }
    }

    fn scan_folders(&mut self) {
        self.parquet_files.clear();
        self.selected_files.clear();

        for folder in &self.folders {
            for entry in WalkDir::new(folder)
                .follow_links(true)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if ext.eq_ignore_ascii_case("parquet") {
                            // Calculate relative path from the folder
                            let display_path = path
                                .strip_prefix(folder)
                                .map(|p| p.to_string_lossy().to_string())
                                .unwrap_or_else(|_| {
                                    path.file_name()
                                        .map(|n| n.to_string_lossy().to_string())
                                        .unwrap_or_else(|| path.to_string_lossy().to_string())
                                });

                            self.parquet_files.push(ParquetFile {
                                full_path: path.to_path_buf(),
                                display_path,
                            });
                        }
                    }
                }
            }
        }

        self.parquet_files.sort_by(|a, b| a.display_path.cmp(&b.display_path));
        self.status_message = format!("Found {} parquet file(s).", self.parquet_files.len());
    }

    fn select_all(&mut self) {
        for i in 0..self.parquet_files.len() {
            self.selected_files.insert(i);
        }
    }

    fn deselect_all(&mut self) {
        self.selected_files.clear();
    }

    fn merge_selected(&mut self) {
        if self.selected_files.is_empty() {
            self.status_message = "No files selected for merging.".to_string();
            return;
        }

        // Show save dialog
        let save_path = rfd::FileDialog::new()
            .add_filter("Parquet", &["parquet"])
            .set_file_name("merged.parquet")
            .save_file();

        let Some(output_path) = save_path else {
            self.status_message = "Merge cancelled.".to_string();
            return;
        };

        self.is_processing = true;
        self.status_message = "Merging files...".to_string();

        // Collect selected file paths (using full paths)
        let files_to_merge: Vec<PathBuf> = self
            .selected_files
            .iter()
            .filter_map(|&i| self.parquet_files.get(i).map(|f| f.full_path.clone()))
            .collect();

        match merge_parquet_files(&files_to_merge, &output_path) {
            Ok(row_count) => {
                let mut status = format!(
                    "Successfully merged {} files ({} total rows) to: {}",
                    files_to_merge.len(),
                    row_count,
                    output_path.display()
                );

                // Export to CSV if checkbox is checked
                if self.export_to_csv {
                    let csv_path = output_path.with_extension("csv");
                    match export_parquet_to_csv(&output_path, &csv_path) {
                        Ok(()) => {
                            status.push_str(&format!(" | CSV exported to: {}", csv_path.display()));
                        }
                        Err(e) => {
                            status.push_str(&format!(" | CSV export failed: {}", e));
                        }
                    }
                }

                self.status_message = status;
            }
            Err(e) => {
                self.status_message = format!("Error merging files: {}", e);
            }
        }

        self.is_processing = false;
    }
}

fn merge_parquet_files(files: &[PathBuf], output_path: &PathBuf) -> Result<usize, Box<dyn std::error::Error>> {
    if files.is_empty() {
        return Err("No files to merge".into());
    }

    // Read the first file to get the schema
    let first_file = std::fs::File::open(&files[0])?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(first_file)?;
    let schema = builder.schema().clone();

    // Collect all record batches from all files
    let mut all_batches: Vec<RecordBatch> = Vec::new();

    for file_path in files {
        let file = std::fs::File::open(file_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

        // Check schema compatibility
        let file_schema = builder.schema();
        if !schemas_compatible(&schema, file_schema) {
            return Err(format!(
                "Schema mismatch in file: {}. Expected schema to match the first file.",
                file_path.display()
            ).into());
        }

        let reader = builder.build()?;
        for batch_result in reader {
            let batch = batch_result?;
            all_batches.push(batch);
        }
    }

    // Count total rows
    let total_rows: usize = all_batches.iter().map(|b| b.num_rows()).sum();

    // Write all batches to output file
    let output_file = std::fs::File::create(output_path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(output_file, schema, Some(props))?;

    for batch in all_batches {
        writer.write(&batch)?;
    }

    writer.close()?;

    Ok(total_rows)
}

fn schemas_compatible(schema1: &Arc<Schema>, schema2: &Arc<Schema>) -> bool {
    if schema1.fields().len() != schema2.fields().len() {
        return false;
    }

    for (f1, f2) in schema1.fields().iter().zip(schema2.fields().iter()) {
        if f1.name() != f2.name() || f1.data_type() != f2.data_type() {
            return false;
        }
    }

    true
}

fn export_parquet_to_csv(parquet_path: &PathBuf, csv_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::open(parquet_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let reader = builder.build()?;

    let mut output = std::fs::File::create(csv_path)?;

    // Write header row
    let headers: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| escape_csv_field(f.name()))
        .collect();
    writeln!(output, "{}", headers.join(","))?;

    // Write data rows
    for batch_result in reader {
        let batch = batch_result?;

        for row_idx in 0..batch.num_rows() {
            let row: Vec<String> = batch
                .columns()
                .iter()
                .map(|col| get_cell_value_as_string(col, row_idx))
                .collect();
            writeln!(output, "{}", row.join(","))?;
        }
    }

    Ok(())
}

fn escape_csv_field(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn get_cell_value_as_string(array: &Arc<dyn Array>, idx: usize) -> String {
    use arrow::array::*;

    if array.is_null(idx) {
        return String::new();
    }

    let value = match array.data_type() {
        DataType::Boolean => {
            let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            arr.value(idx).to_string()
        }
        DataType::Int8 => {
            let arr = array.as_any().downcast_ref::<Int8Array>().unwrap();
            arr.value(idx).to_string()
        }
        DataType::Int16 => {
            let arr = array.as_any().downcast_ref::<Int16Array>().unwrap();
            arr.value(idx).to_string()
        }
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
            arr.value(idx).to_string()
        }
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            arr.value(idx).to_string()
        }
        DataType::UInt8 => {
            let arr = array.as_any().downcast_ref::<UInt8Array>().unwrap();
            arr.value(idx).to_string()
        }
        DataType::UInt16 => {
            let arr = array.as_any().downcast_ref::<UInt16Array>().unwrap();
            arr.value(idx).to_string()
        }
        DataType::UInt32 => {
            let arr = array.as_any().downcast_ref::<UInt32Array>().unwrap();
            arr.value(idx).to_string()
        }
        DataType::UInt64 => {
            let arr = array.as_any().downcast_ref::<UInt64Array>().unwrap();
            arr.value(idx).to_string()
        }
        DataType::Float32 => {
            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
            arr.value(idx).to_string()
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            arr.value(idx).to_string()
        }
        DataType::Utf8 => {
            let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
            arr.value(idx).to_string()
        }
        DataType::LargeUtf8 => {
            let arr = array.as_any().downcast_ref::<LargeStringArray>().unwrap();
            arr.value(idx).to_string()
        }
        DataType::Date32 => {
            let arr = array.as_any().downcast_ref::<Date32Array>().unwrap();
            format!("{:?}", arr.value_as_date(idx))
        }
        DataType::Date64 => {
            let arr = array.as_any().downcast_ref::<Date64Array>().unwrap();
            format!("{:?}", arr.value_as_datetime(idx))
        }
        DataType::Timestamp(_, _) => {
            if let Some(arr) = array.as_any().downcast_ref::<TimestampMicrosecondArray>() {
                format!("{:?}", arr.value_as_datetime(idx))
            } else if let Some(arr) = array.as_any().downcast_ref::<TimestampMillisecondArray>() {
                format!("{:?}", arr.value_as_datetime(idx))
            } else if let Some(arr) = array.as_any().downcast_ref::<TimestampSecondArray>() {
                format!("{:?}", arr.value_as_datetime(idx))
            } else if let Some(arr) = array.as_any().downcast_ref::<TimestampNanosecondArray>() {
                format!("{:?}", arr.value_as_datetime(idx))
            } else {
                format!("{:?}", array.slice(idx, 1))
            }
        }
        _ => {
            format!("{:?}", array.slice(idx, 1))
        }
    };

    escape_csv_field(&value)
}

impl eframe::App for ParquetMergerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.add_space(8.0);
            ui.heading("Parquet File Merger");
            ui.add_space(4.0);
        });

        egui::TopBottomPanel::bottom("bottom_panel").show(ctx, |ui| {
            ui.add_space(8.0);
            if !self.status_message.is_empty() {
                ui.label(&self.status_message);
            }
            ui.add_space(8.0);
        });

        egui::SidePanel::left("folders_panel")
            .resizable(true)
            .default_width(250.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                ui.heading("Folders");
                ui.add_space(8.0);

                ui.horizontal(|ui| {
                    if ui.button("Add Folder").clicked() && !self.is_processing {
                        self.add_folder();
                    }
                    if ui.button("Scan").clicked() && !self.is_processing && !self.folders.is_empty() {
                        self.scan_folders();
                    }
                });

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);

                let mut folder_to_remove: Option<usize> = None;

                egui::ScrollArea::vertical()
                    .id_salt("folders_scroll")
                    .show(ui, |ui| {
                        for (i, folder) in self.folders.iter().enumerate() {
                            ui.horizontal(|ui| {
                                if ui.small_button("X").clicked() && !self.is_processing {
                                    folder_to_remove = Some(i);
                                }
                                let folder_name = folder
                                    .file_name()
                                    .map(|n| n.to_string_lossy().to_string())
                                    .unwrap_or_else(|| folder.to_string_lossy().to_string());
                                ui.label(&folder_name).on_hover_text(folder.to_string_lossy());
                            });
                        }
                    });

                if let Some(idx) = folder_to_remove {
                    self.remove_folder(idx);
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.heading("Parquet Files");
                ui.add_space(16.0);
                ui.label(format!("{} files found", self.parquet_files.len()));
            });
            ui.add_space(8.0);

            ui.horizontal(|ui| {
                if ui.button("Select All").clicked() && !self.is_processing {
                    self.select_all();
                }
                if ui.button("Deselect All").clicked() && !self.is_processing {
                    self.deselect_all();
                }
                ui.add_space(16.0);
                let selected_count = self.selected_files.len();
                ui.label(format!("{} selected", selected_count));
                ui.add_space(16.0);
                if ui
                    .add_enabled(
                        !self.is_processing && selected_count > 0,
                        egui::Button::new("Merge Selected"),
                    )
                    .clicked()
                {
                    self.merge_selected();
                }
                ui.add_space(16.0);
                ui.checkbox(&mut self.export_to_csv, "Also export to CSV");
            });

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(8.0);

            egui::ScrollArea::vertical()
                .id_salt("files_scroll")
                .show(ui, |ui| {
                    for (i, pq_file) in self.parquet_files.iter().enumerate() {
                        let mut is_selected = self.selected_files.contains(&i);

                        ui.horizontal(|ui| {
                            if ui.checkbox(&mut is_selected, "").changed() {
                                if is_selected {
                                    self.selected_files.insert(i);
                                } else {
                                    self.selected_files.remove(&i);
                                }
                            }
                            // Display relative path, hover shows full path
                            ui.label(&pq_file.display_path)
                                .on_hover_text(pq_file.full_path.to_string_lossy());
                        });
                    }
                });
        });
    }
}
