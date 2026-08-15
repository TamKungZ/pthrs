use std::{env, process::ExitCode};

use pthrs::{PthArchive, TensorMeta, TensorReadBuffer};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        return Err(usage());
    };
    if path == "-h" || path == "--help" {
        println!("{}", usage());
        return Ok(());
    }
    let mut list = false;
    let mut validate = false;
    let mut tensor_name = None;
    let mut values = 8usize;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--list" => list = true,
            "--validate" => validate = true,
            "--tensor" => tensor_name = Some(args.next().ok_or("--tensor requires a name")?),
            "--values" => {
                values = args
                    .next()
                    .ok_or("--values requires a number")?
                    .parse()
                    .map_err(|_| "--values must be a non-negative integer")?
            }
            "-h" | "--help" => {
                println!("{}", usage());
                return Ok(());
            }
            unknown => return Err(format!("unknown option: {unknown}\n\n{}", usage())),
        }
    }

    let mut archive = PthArchive::open(&path).map_err(|error| error.to_string())?;
    let checkpoint = archive.checkpoint();
    println!("file: {path}");
    println!("pickle protocol: {}", checkpoint.pickle_protocol());
    println!(
        "serialization version: {}",
        archive.serialization_version().unwrap_or("unknown")
    );
    println!("byte order: {:?}", archive.byte_order());
    println!("tensors: {}", checkpoint.tensor_count());
    println!("metadata:");
    for (key, value) in checkpoint.metadata() {
        println!("  {key}: {}", value.pretty());
    }

    if validate {
        let summary = checkpoint.summary().map_err(|error| error.to_string())?;
        println!("summary:");
        println!("  storages: {}", summary.storage_count);
        println!("  tensor elements: {}", summary.tensor_elements);
        println!("  logical tensor bytes: {}", summary.logical_tensor_bytes);
        println!("  storage bytes: {}", summary.storage_bytes);
        match checkpoint.voice_model_info() {
            Ok(info) => {
                println!("voice model:");
                println!("  sample rate: {}", info.config.sample_rate);
                println!("  speakers: {}", info.config.speaker_count);
                println!("  phone channels: {:?}", info.phone_feature_channels);
                println!("  pitch guidance: {}", info.pitch_guidance);
                let report = info.validate(checkpoint);
                println!("  valid: {}", report.is_valid());
                for error in report.errors {
                    println!("  error: {error}");
                }
                for warning in report.warnings {
                    println!("  warning: {warning}");
                }
            }
            Err(error) => println!("voice model metadata: {error}"),
        }
    }

    if list {
        println!("tensor list:");
        for (name, tensor) in archive.checkpoint().tensors() {
            println!("  {name}: {}", describe(tensor));
        }
    }
    if let Some(name) = tensor_name {
        let mut buffer = TensorReadBuffer::new();
        let tensor = archive
            .read_tensor_into(&name, &mut buffer)
            .map_err(|error| error.to_string())?;
        println!("tensor {name}: {}", describe(&tensor.meta));
        if values > 0 {
            let decoded = tensor.to_f32_vec().map_err(|error| error.to_string())?;
            let shown = decoded.len().min(values);
            println!("first {shown} values: {:?}", &decoded[..shown]);
        }
    }
    Ok(())
}

fn describe(tensor: &TensorMeta) -> String {
    format!(
        "{:?} shape={:?} stride={:?} storage={} offset={}{}",
        tensor.dtype,
        tensor.shape,
        tensor.stride,
        tensor.storage.key,
        tensor.storage_offset,
        if tensor.is_contiguous() {
            ""
        } else {
            " non-contiguous"
        },
    )
}

fn usage() -> String {
    "Usage: pthrs-inspect <model.pth> [--list] [--validate] [--tensor NAME] [--values N]\n\
     \n\
     Options:\n\
       --list          List every tensor and its metadata\n\
       --validate      Validate storage and voice-model metadata\n\
       --tensor NAME   Read one tensor's data lazily\n\
       --values N      Print the first N values as f32 (default: 8)\n\
       -h, --help      Show this help"
        .to_owned()
}
