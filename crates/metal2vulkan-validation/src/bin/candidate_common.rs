use metal2vulkan_validation::candidate::{engine, exit_status};
use metal2vulkan_validation::check::admission::ExecutorError;
use metal2vulkan_validation::gpu::wait::{parse_wait_timeout, set_wait_timeout};
use metal2vulkan_validation::gpu::DeviceType;
use metal2vulkan_validation::source::corpus_root;
use metal2vulkan_validation::store::CorpusStore;
use metal2vulkan_validation::worker::alloc::CountingAllocator;
use metal2vulkan_validation::worker::compile::{serve_if_requested, CompileWorker};
use metal2vulkan_validation::worker::device;
use std::path::PathBuf;

// The engine compiles in a killable compile worker (this executable), whose in-child budget is
// this allocator's (worker recipe step 2), and runs the device half of a case in a device worker
// (this executable too).
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

pub fn main() {
    serve_if_requested();
    device::serve_if_requested();
    match run() {
        Ok(()) => {}
        Err(ExecutorError::Refused(refused)) => {
            println!("{refused}");
            std::process::exit(exit_status::REFUSED);
        }
        Err(ExecutorError::NotSupported(not_supported)) => {
            println!("{not_supported}");
            std::process::exit(exit_status::NOT_SUPPORTED);
        }
        Err(ExecutorError::DeviceCrash(crash)) => {
            println!("{crash}");
            std::process::exit(exit_status::DEVICE_CRASH);
        }
        Err(ExecutorError::NeedsMetal(needs)) => {
            println!("{needs}");
            std::process::exit(exit_status::NEEDS_METAL);
        }
        Err(ExecutorError::VenueMismatch(mismatch)) => {
            println!("{mismatch}");
            std::process::exit(exit_status::VENUE_MISMATCH);
        }
        Err(ExecutorError::Failed(error)) => {
            eprintln!("corpus-vulkan: {error}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<(), ExecutorError> {
    let mut root = corpus_root();
    let mut case_id = None;
    let mut air_sha256 = None;
    let mut metal_environment_id = None;
    let mut environment_id = None;
    let mut device_type = None;
    let mut profiles_layer = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--corpus" => root = PathBuf::from(required(&mut args, "--corpus")?),
            "--private-corpus" => metal2vulkan_validation::source::use_private_corpus_root(
                PathBuf::from(required(&mut args, "--private-corpus")?),
            )?,
            "--case-id" => case_id = Some(required(&mut args, "--case-id")?),
            "--air-sha256" => air_sha256 = Some(required(&mut args, "--air-sha256")?),
            "--source-index" => metal2vulkan_validation::source::use_source_index(PathBuf::from(
                required(&mut args, "--source-index")?,
            ))?,
            "--metal-environment-id" => {
                metal_environment_id = Some(required(&mut args, "--metal-environment-id")?)
            }
            "--environment-id" => environment_id = Some(required(&mut args, "--environment-id")?),
            "--device-type" => {
                device_type = Some(DeviceType::parse(&required(&mut args, "--device-type")?)?)
            }
            "--profiles-layer" => profiles_layer = Some(required(&mut args, "--profiles-layer")?),
            "--gpu-wait-timeout" => set_wait_timeout(parse_wait_timeout(&required(
                &mut args,
                "--gpu-wait-timeout",
            )?)?)?,
            "-h" | "--help" => {
                println!(
                    "usage: corpus-vulkan [--corpus DIR] [--private-corpus DIR] --case-id HASH --metal-environment-id ID --environment-id ID --device-type cpu|gpu \\\n\
                     [--air-sha256 HASH] [--source-index FILE] \\\n\
                     [--gpu-wait-timeout SECONDS] [--profiles-layer P]\n\
                     cpu matches only a CPU Vulkan implementation.\n\
                     --air-sha256 HASH, the case's AIR, reads the case from only the case shard HASH\n\
                     selects (refused if the case there is over another AIR); without it every case\n\
                     shard is read. The case's Metal row is read from its own Metal shard either way.\n\
                     --source-index FILE reads private sources through that corpus index's byte\n\
                     locations, read only: a location that is missing, stale, or not the source falls\n\
                     back to parsing the source's shard.\n\
                     It runs the engine: it compiles in a killable worker (20 s, 500 MiB), binds,\n\
                     and executes through the engine's binder under VK_LAYER_KHRONOS_validation, stores the\n\
                     row in the ID@engine slot, runs a cyclic case as compiled on any device\n\
                     (no execution bound; a shader that does not end is not cancelled: the finite\n\
                     device wait ends only this process's wait and the hang tripwire only refuses\n\
                     later runs, so only the OS or driver GPU timeout stops it), and reports a\n\
                     case it cannot run as not_supported{{...}} (exit 4). The device half runs\n\
                     in a child of this executable; a child a signal ends is\n\
                     device_crash{{SIGNAL}} (exit 5).\n\
                     --profiles-layer P: run on the device restricted to the committed\n\
                     profile validation/profiles/P.json through VK_LAYER_KHRONOS_profiles, after\n\
                     checking that the device has P and reports exactly P; the row records the layer\n\
                     and P and is stored in the ID@engine@P slot.\n\
                     It enables every robustness feature the device's description (or P)\n\
                     states (robustBufferAccess, robustBufferAccess2, robustImageAccess,\n\
                     robustImageAccess2), so the product compiles the robust containment encodings,\n\
                     and a case is not_supported{{not_enabled}} (exit 4) on a device that cannot enable\n\
                     one it states.\n\
                     The executor compares only against the case's Metal observation in the\n\
                     --metal-environment-id environment recorded by the current oracle for the\n\
                     checked case; without one it reports needs_metal_qualification{{...}} (exit 6)\n\
                     before any device opens."
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument {arg:?}").into()),
        }
    }
    let case_id = case_id.ok_or("--case-id is required")?;
    let metal_environment_id = metal_environment_id.ok_or("--metal-environment-id is required")?;
    let environment_id = environment_id.ok_or("--environment-id is required")?;
    let device_type = device_type.ok_or("--device-type is required")?;
    let store = CorpusStore::new(&root);
    let case = match &air_sha256 {
        Some(air_sha256) => store.find_case_of_air(&case_id, air_sha256)?,
        None => store.find_case(&case_id)?,
    }
    .ok_or_else(|| format!("case_id {case_id} not found"))?;
    if let Some(profile) = &profiles_layer {
        metal2vulkan_validation::candidate::restrict_devices_to(profile)?;
    }
    let observation = engine::execute(
        &root,
        case,
        device_type,
        &metal_environment_id,
        &environment_id,
        &CompileWorker::current()?,
    )?;
    println!(
        "{:?} {} candidate_output_sha256={}",
        observation.status, observation.case_id, observation.candidate_output_sha256
    );
    if let Some(withheld) = observation.environment.get("withheld_validation_messages") {
        println!("withheld validation messages: {withheld}");
    }
    if observation.status == metal2vulkan_validation::observation::CandidateStatus::Mismatch {
        return Err("candidate output does not match the exact Metal observation".into());
    }
    Ok(())
}

fn required(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}
