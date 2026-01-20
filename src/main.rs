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
            .with_inner_size([1100.0, 600.0])
            .with_min_inner_size([900.0, 400.0]),
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

/// A batch of files to be merged together
#[derive(Clone)]
struct Batch {
    /// Indices of files in this batch (references parquet_files)
    file_indices: Vec<usize>,
    /// Name of the batch (used for output filename)
    name: String,
    /// Whether there's a schema mismatch between files
    has_schema_mismatch: bool,
}

/// Progress state for batch processing
#[derive(Default)]
struct MergeProgress {
    /// Whether merge is in progress
    is_merging: bool,
    /// Current batch being processed (0-indexed)
    current_batch: usize,
    /// Total number of batches
    total_batches: usize,
    /// Progress message
    message: String,
}

struct ParquetMergerApp {
    /// List of folders to scan
    folders: Vec<PathBuf>,
    /// Found parquet files after scanning
    parquet_files: Vec<ParquetFile>,
    /// Set of selected file indices for adding to batch
    selected_files: HashSet<usize>,
    /// List of batches to process
    batches: Vec<Batch>,
    /// Status message to display
    status_message: String,
    /// Whether an operation is in progress
    is_processing: bool,
    /// Whether to also export to CSV
    export_to_csv: bool,
    /// Merge progress tracking
    merge_progress: MergeProgress,
    /// Search filter for files
    search_filter: String,
    /// Auto-remove batches after successful merge
    auto_remove_completed: bool,
    /// Enable smart batching (auto-group files by filename)
    smart_batch_enabled: bool,
}

impl Default for ParquetMergerApp {
    fn default() -> Self {
        Self {
            folders: Vec::new(),
            parquet_files: Vec::new(),
            selected_files: HashSet::new(),
            batches: Vec::new(),
            status_message: String::new(),
            is_processing: false,
            export_to_csv: false,
            merge_progress: MergeProgress::default(),
            search_filter: String::new(),
            auto_remove_completed: true,
            smart_batch_enabled: true, // Default to true
        }
    }
}

impl ParquetMergerApp {
    fn add_folder(&mut self) {
        if let Some(folder) = rfd::FileDialog::new().pick_folder() {
            if !self.folders.contains(&folder) {
                self.folders.push(folder);
                // Auto-scan after adding folder
                self.scan_folders();
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
            self.batches.clear();
            self.status_message = "Folder removed.".to_string();
        }
    }

    fn scan_folders(&mut self) {
        self.parquet_files.clear();
        self.selected_files.clear();
        self.batches.clear();

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

        let file_count = self.parquet_files.len();

        // Auto-run smart batching if enabled
        if self.smart_batch_enabled && file_count > 0 {
            self.smart_batch();
        } else {
            self.status_message = format!("Found {} parquet file(s).", file_count);
        }
    }

    fn add_batch(&mut self) {
        if self.selected_files.is_empty() {
            self.status_message = "No files selected. Select files first, then click '>'.".to_string();
            return;
        }

        let mut file_indices: Vec<usize> = self.selected_files.iter().cloned().collect();
        file_indices.sort();

        // Get file paths for schema check and name generation
        let file_paths: Vec<&PathBuf> = file_indices
            .iter()
            .filter_map(|&i| self.parquet_files.get(i).map(|f| &f.full_path))
            .collect();

        let file_names: Vec<&str> = file_indices
            .iter()
            .filter_map(|&i| {
                self.parquet_files.get(i).and_then(|f| {
                    f.full_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                })
            })
            .collect();

        // Check for schema mismatch
        let has_schema_mismatch = check_schema_mismatch(&file_paths);

        // Generate batch name from common parts of file names
        let name = generate_batch_name(&file_names, self.batches.len() + 1);

        self.batches.push(Batch {
            file_indices,
            name: name.clone(),
            has_schema_mismatch,
        });
        self.selected_files.clear();

        if has_schema_mismatch {
            self.status_message = format!("Created batch '{}' (WARNING: schema mismatch detected!).", name);
        } else {
            self.status_message = format!("Created batch '{}'.", name);
        }
    }

    fn remove_batch(&mut self, index: usize) {
        if index < self.batches.len() {
            self.batches.remove(index);
            self.status_message = "Batch removed.".to_string();
        }
    }

    fn smart_batch(&mut self) {
        use std::collections::HashMap;

        if self.parquet_files.is_empty() {
            self.status_message = "No files to batch. Add a folder and scan first.".to_string();
            return;
        }

        // Group files by their filename (ignoring path)
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();

        for (idx, pq_file) in self.parquet_files.iter().enumerate() {
            let filename = pq_file.full_path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();

            groups.entry(filename).or_default().push(idx);
        }

        // Create batches for groups with more than one file
        let mut batches_created = 0;
        let mut single_files = 0;

        for (filename, file_indices) in groups {
            if file_indices.len() > 1 {
                // Get file paths for schema check
                let file_paths: Vec<&PathBuf> = file_indices
                    .iter()
                    .filter_map(|&i| self.parquet_files.get(i).map(|f| &f.full_path))
                    .collect();

                let has_schema_mismatch = check_schema_mismatch(&file_paths);

                // Use filename without extension as batch name
                let name = std::path::Path::new(&filename)
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| format!("batch_{}", self.batches.len() + 1));

                self.batches.push(Batch {
                    file_indices,
                    name,
                    has_schema_mismatch,
                });
                batches_created += 1;
            } else {
                single_files += 1;
            }
        }

        if batches_created > 0 {
            self.status_message = format!(
                "Smart batching: created {} batch(es). {} file(s) had no matches.",
                batches_created, single_files
            );
        } else {
            self.status_message = "No files with matching names found across different paths.".to_string();
        }
    }

    fn merge_batches(&mut self) {
        if self.batches.is_empty() {
            self.status_message = "No batches to merge. Add batches first using '>'.".to_string();
            return;
        }

        // Ask for output folder
        let output_folder = rfd::FileDialog::new()
            .set_title("Select output folder for merged files")
            .pick_folder();

        let Some(output_dir) = output_folder else {
            self.status_message = "Merge cancelled.".to_string();
            return;
        };

        self.is_processing = true;
        self.merge_progress = MergeProgress {
            is_merging: true,
            current_batch: 0,
            total_batches: self.batches.len(),
            message: "Starting merge...".to_string(),
        };

        // Create "merged" subdirectory
        let out_dir = output_dir.join("merged");
        if let Err(e) = std::fs::create_dir_all(&out_dir) {
            self.status_message = format!("Failed to create output directory: {}", e);
            self.is_processing = false;
            self.merge_progress.is_merging = false;
            return;
        }

        let mut success_count = 0;
        let mut error_messages: Vec<String> = Vec::new();
        let mut successful_batch_indices: Vec<usize> = Vec::new();

        for (batch_idx, batch) in self.batches.iter().enumerate() {
            self.merge_progress.current_batch = batch_idx + 1;
            self.merge_progress.message = format!("Processing '{}'...", batch.name);

            let files_to_merge: Vec<PathBuf> = batch
                .file_indices
                .iter()
                .filter_map(|&i| self.parquet_files.get(i).map(|f| f.full_path.clone()))
                .collect();

            if files_to_merge.is_empty() {
                error_messages.push(format!("'{}': No valid files", batch.name));
                continue;
            }

            // Sanitize batch name for filename
            let safe_name = sanitize_filename(&batch.name);
            let output_path = out_dir.join(format!("{}.parquet", safe_name));

            match merge_parquet_files(&files_to_merge, &output_path) {
                Ok(row_count) => {
                    success_count += 1;
                    successful_batch_indices.push(batch_idx);

                    // Export to CSV if checkbox is checked
                    if self.export_to_csv {
                        let csv_path = output_path.with_extension("csv");
                        if let Err(e) = export_parquet_to_csv(&output_path, &csv_path) {
                            error_messages.push(format!("'{}' CSV export failed: {}", batch.name, e));
                        }
                    }

                    self.merge_progress.message = format!(
                        "'{}' completed ({} rows)",
                        batch.name,
                        row_count
                    );
                }
                Err(e) => {
                    error_messages.push(format!("'{}': {}", batch.name, e));
                }
            }
        }

        // Remove successful batches if auto-remove is enabled
        if self.auto_remove_completed {
            // Remove in reverse order to maintain correct indices
            for &idx in successful_batch_indices.iter().rev() {
                self.batches.remove(idx);
            }
        }

        self.merge_progress.is_merging = false;
        self.is_processing = false;

        if error_messages.is_empty() {
            self.status_message = format!(
                "Successfully merged {} batch(es) to: {}",
                success_count,
                out_dir.display()
            );
        } else {
            self.status_message = format!(
                "Merged {} batch(es), {} error(s): {}",
                success_count,
                error_messages.len(),
                error_messages.join("; ")
            );
        }
    }
}

/// Check if there's a schema mismatch between files
fn check_schema_mismatch(files: &[&PathBuf]) -> bool {
    if files.len() < 2 {
        return false;
    }

    let first_schema = match get_file_schema(files[0]) {
        Some(s) => s,
        None => return true, // Can't read schema, assume mismatch
    };

    for file in files.iter().skip(1) {
        match get_file_schema(file) {
            Some(schema) => {
                if !schemas_compatible(&first_schema, &schema) {
                    return true;
                }
            }
            None => return true, // Can't read schema, assume mismatch
        }
    }

    false
}

/// Get schema from a parquet file
fn get_file_schema(path: &PathBuf) -> Option<Arc<Schema>> {
    let file = std::fs::File::open(path).ok()?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).ok()?;
    Some(builder.schema().clone())
}

/// Generate a batch name from common parts of file names
fn generate_batch_name(file_names: &[&str], batch_number: usize) -> String {
    if file_names.is_empty() {
        return format!("batch_{}", batch_number);
    }

    if file_names.len() == 1 {
        return file_names[0].to_string();
    }

    // Try to find common prefix
    let common_prefix = find_common_prefix(file_names);
    if !common_prefix.is_empty() && common_prefix.len() >= 3 {
        // Clean up trailing underscores, hyphens, or numbers
        let cleaned = common_prefix
            .trim_end_matches(|c: char| c == '_' || c == '-' || c.is_ascii_digit())
            .trim_end_matches(|c: char| c == '_' || c == '-');
        if !cleaned.is_empty() && cleaned.len() >= 3 {
            return cleaned.to_string();
        }
    }

    // Try to find common suffix
    let common_suffix = find_common_suffix(file_names);
    if !common_suffix.is_empty() && common_suffix.len() >= 3 {
        let cleaned = common_suffix
            .trim_start_matches(|c: char| c == '_' || c == '-' || c.is_ascii_digit())
            .trim_start_matches(|c: char| c == '_' || c == '-');
        if !cleaned.is_empty() && cleaned.len() >= 3 {
            return cleaned.to_string();
        }
    }

    // Try to find common substring
    if let Some(common) = find_common_substring(file_names) {
        if common.len() >= 3 {
            return common;
        }
    }

    format!("batch_{}", batch_number)
}

fn find_common_prefix(strings: &[&str]) -> String {
    if strings.is_empty() {
        return String::new();
    }

    let first = strings[0];
    let mut prefix_len = first.len();

    for s in strings.iter().skip(1) {
        prefix_len = first
            .chars()
            .zip(s.chars())
            .take(prefix_len)
            .take_while(|(a, b)| a == b)
            .count();
    }

    first.chars().take(prefix_len).collect()
}

fn find_common_suffix(strings: &[&str]) -> String {
    if strings.is_empty() {
        return String::new();
    }

    let reversed: Vec<String> = strings.iter().map(|s| s.chars().rev().collect()).collect();
    let refs: Vec<&str> = reversed.iter().map(|s| s.as_str()).collect();
    find_common_prefix(&refs).chars().rev().collect()
}

fn find_common_substring(strings: &[&str]) -> Option<String> {
    if strings.is_empty() || strings[0].is_empty() {
        return None;
    }

    let first = strings[0];

    // Try substrings of decreasing length
    for len in (3..=first.len()).rev() {
        for start in 0..=(first.len() - len) {
            let substring = &first[start..start + len];
            // Skip if mostly numbers or special chars
            if substring.chars().filter(|c| c.is_alphabetic()).count() < 2 {
                continue;
            }
            if strings.iter().all(|s| s.contains(substring)) {
                return Some(substring.to_string());
            }
        }
    }

    None
}

/// Sanitize a string for use as a filename
fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn merge_parquet_files(files: &[PathBuf], output_path: &PathBuf) -> Result<usize, Box<dyn std::error::Error>> {
    if files.is_empty() {
        return Err("No files to merge".into());
    }

    // Collect all schemas first
    let mut schemas: Vec<Arc<Schema>> = Vec::new();
    for file_path in files {
        if let Some(schema) = get_file_schema(file_path) {
            schemas.push(schema);
        } else {
            return Err(format!("Cannot read schema from: {}", file_path.display()).into());
        }
    }

    // Check if all schemas are compatible
    let all_compatible = schemas.windows(2).all(|w| schemas_compatible(&w[0], &w[1]));

    let (output_schema, common_columns) = if all_compatible {
        // All schemas match, use the first one
        (schemas[0].clone(), None)
    } else {
        // Find common columns across all schemas
        let common = find_common_columns(&schemas);
        if common.is_empty() {
            return Err("No common columns found across all files".into());
        }
        let new_schema = create_schema_from_columns(&schemas[0], &common);
        (Arc::new(new_schema), Some(common))
    };

    let mut all_batches: Vec<RecordBatch> = Vec::new();

    for file_path in files {
        let file = std::fs::File::open(file_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;

        for batch_result in reader {
            let batch = batch_result?;

            // Project to common columns if needed
            let projected_batch = if let Some(ref cols) = common_columns {
                project_batch_to_columns(&batch, cols)?
            } else {
                batch
            };

            all_batches.push(projected_batch);
        }
    }

    let total_rows: usize = all_batches.iter().map(|b| b.num_rows()).sum();

    let output_file = std::fs::File::create(output_path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(output_file, output_schema, Some(props))?;

    for batch in all_batches {
        writer.write(&batch)?;
    }

    writer.close()?;

    Ok(total_rows)
}

/// Find columns that exist in all schemas with compatible types
fn find_common_columns(schemas: &[Arc<Schema>]) -> Vec<String> {
    if schemas.is_empty() {
        return Vec::new();
    }

    let first_schema = &schemas[0];
    let mut common: Vec<String> = Vec::new();

    for field in first_schema.fields() {
        let name = field.name();
        let data_type = field.data_type();

        // Check if this column exists in all other schemas with the same type
        let exists_in_all = schemas.iter().skip(1).all(|schema| {
            schema.field_with_name(name)
                .map(|f| f.data_type() == data_type)
                .unwrap_or(false)
        });

        if exists_in_all {
            common.push(name.clone());
        }
    }

    common
}

/// Create a new schema with only the specified columns
fn create_schema_from_columns(original: &Arc<Schema>, columns: &[String]) -> Schema {
    use arrow::datatypes::Field;

    let fields: Vec<Arc<Field>> = columns
        .iter()
        .filter_map(|name| original.field_with_name(name).ok().map(|f| Arc::new(f.clone())))
        .collect();

    Schema::new(fields)
}

/// Project a record batch to only include specified columns
fn project_batch_to_columns(batch: &RecordBatch, columns: &[String]) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let schema = batch.schema();
    let indices: Vec<usize> = columns
        .iter()
        .filter_map(|name| schema.index_of(name).ok())
        .collect();

    let projected_columns: Vec<Arc<dyn Array>> = indices
        .iter()
        .map(|&i| batch.column(i).clone())
        .collect();

    let new_schema = create_schema_from_columns(&schema, columns);

    Ok(RecordBatch::try_new(Arc::new(new_schema), projected_columns)?)
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

/// Check if a column name is an internal/index column that should be excluded
fn is_internal_column(name: &str) -> bool {
    name.starts_with("__") && name.ends_with("__")
}

fn export_parquet_to_csv(parquet_path: &PathBuf, csv_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::open(parquet_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let reader = builder.build()?;

    let mut output = std::fs::File::create(csv_path)?;

    // Get indices of columns to include (excluding internal columns like __index_level_0__)
    let column_indices: Vec<usize> = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, f)| !is_internal_column(f.name()))
        .map(|(i, _)| i)
        .collect();

    let headers: Vec<String> = column_indices
        .iter()
        .map(|&i| escape_csv_field(schema.field(i).name()))
        .collect();
    writeln!(output, "{}", headers.join(","))?;

    for batch_result in reader {
        let batch = batch_result?;

        for row_idx in 0..batch.num_rows() {
            let row: Vec<String> = column_indices
                .iter()
                .map(|&col_idx| get_cell_value_as_string(batch.column(col_idx), row_idx))
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

            // Show progress bar if merging
            if self.merge_progress.is_merging {
                ui.horizontal(|ui| {
                    let progress = self.merge_progress.current_batch as f32 / self.merge_progress.total_batches as f32;
                    let progress_bar = egui::ProgressBar::new(progress)
                        .text(format!(
                            "{} ({}/{})",
                            self.merge_progress.message,
                            self.merge_progress.current_batch,
                            self.merge_progress.total_batches
                        ))
                        .animate(true);
                    ui.add(progress_bar);
                });
            } else if !self.status_message.is_empty() {
                ui.label(&self.status_message);
            }

            ui.add_space(8.0);
        });

        // Left panel - Folders
        egui::SidePanel::left("folders_panel")
            .resizable(true)
            .default_width(200.0)
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

                ui.checkbox(&mut self.smart_batch_enabled, "Smart batch")
                    .on_hover_text("Auto-group files with same name from different paths");

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

        // Right panel - Batches
        egui::SidePanel::right("batches_panel")
            .resizable(true)
            .default_width(280.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.heading("Batches");
                    ui.add_space(8.0);
                    ui.label(format!("({} total)", self.batches.len()));
                });
                ui.add_space(8.0);

                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            !self.is_processing && !self.batches.is_empty(),
                            egui::Button::new("Merge Batches"),
                        )
                        .clicked()
                    {
                        self.merge_batches();
                    }
                });
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.export_to_csv, "CSV");
                    ui.checkbox(&mut self.auto_remove_completed, "Auto-remove");
                });

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);

                let mut batch_to_remove: Option<usize> = None;

                egui::ScrollArea::vertical()
                    .id_salt("batches_scroll")
                    .show(ui, |ui| {
                        for batch_idx in 0..self.batches.len() {
                            let batch = &self.batches[batch_idx];
                            let has_mismatch = batch.has_schema_mismatch;
                            let file_count = batch.file_indices.len();

                            ui.group(|ui| {
                                ui.horizontal(|ui| {
                                    // Warning icon for schema mismatch
                                    if has_mismatch {
                                        ui.label(egui::RichText::new("⚠").color(egui::Color32::YELLOW))
                                            .on_hover_text("Schema mismatch detected! Files have different columns or types.");
                                    }

                                    // Editable batch name
                                    let name_response = ui.add(
                                        egui::TextEdit::singleline(&mut self.batches[batch_idx].name)
                                            .desired_width(120.0)
                                            .hint_text("batch name")
                                    );
                                    if name_response.changed() {
                                        // Name was edited
                                    }

                                    ui.label(format!("({})", file_count));

                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        if ui.small_button("X").clicked() && !self.is_processing {
                                            batch_to_remove = Some(batch_idx);
                                        }
                                    });
                                });

                                // Show files in this batch
                                let batch = &self.batches[batch_idx];
                                for &file_idx in &batch.file_indices {
                                    if let Some(pq_file) = self.parquet_files.get(file_idx) {
                                        ui.label(format!("  {}", &pq_file.display_path))
                                            .on_hover_text(pq_file.full_path.to_string_lossy().to_string());
                                    }
                                }
                            });
                            ui.add_space(4.0);
                        }
                    });

                if let Some(idx) = batch_to_remove {
                    self.remove_batch(idx);
                }
            });

        // Center panel - Parquet Files
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.heading("Parquet Files");
                ui.add_space(16.0);
                ui.label(format!("{} files found", self.parquet_files.len()));
            });
            ui.add_space(8.0);

            // Search filter
            ui.horizontal(|ui| {
                ui.label("Search:");
                ui.add(egui::TextEdit::singleline(&mut self.search_filter).hint_text("Filter files..."));
                if ui.small_button("Clear").clicked() {
                    self.search_filter.clear();
                }
            });
            ui.add_space(8.0);

            // Collect filtered file indices
            let search_lower = self.search_filter.to_lowercase();
            let filtered_indices: Vec<usize> = self
                .parquet_files
                .iter()
                .enumerate()
                .filter(|(_, pq_file)| {
                    search_lower.is_empty() || pq_file.display_path.to_lowercase().contains(&search_lower)
                })
                .map(|(i, _)| i)
                .collect();

            ui.horizontal(|ui| {
                if ui.button("Select All").clicked() && !self.is_processing {
                    // Select only filtered files
                    for &i in &filtered_indices {
                        self.selected_files.insert(i);
                    }
                }
                if ui.button("Deselect All").clicked() && !self.is_processing {
                    // Deselect only filtered files
                    for &i in &filtered_indices {
                        self.selected_files.remove(&i);
                    }
                }
                ui.add_space(16.0);
                let selected_count = self.selected_files.len();
                ui.label(format!("{} selected", selected_count));
                if !search_lower.is_empty() {
                    ui.label(format!("({} shown)", filtered_indices.len()));
                }
                ui.add_space(16.0);

                // Add to batch button
                if ui
                    .add_enabled(
                        !self.is_processing && selected_count > 0,
                        egui::Button::new(" > "),
                    )
                    .on_hover_text("Add selected files as a new batch")
                    .clicked()
                {
                    self.add_batch();
                }
            });

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(8.0);

            egui::ScrollArea::vertical()
                .id_salt("files_scroll")
                .show(ui, |ui| {
                    for &i in &filtered_indices {
                        let pq_file = &self.parquet_files[i];
                        let mut is_selected = self.selected_files.contains(&i);

                        ui.horizontal(|ui| {
                            if ui.checkbox(&mut is_selected, "").changed() {
                                if is_selected {
                                    self.selected_files.insert(i);
                                } else {
                                    self.selected_files.remove(&i);
                                }
                            }
                            ui.label(&pq_file.display_path)
                                .on_hover_text(pq_file.full_path.to_string_lossy());
                        });
                    }
                });
        });
    }
}
