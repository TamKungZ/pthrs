use pthrs::{
    FaissIvfFlatIndex, LoadedIvfFlatIndex, PthArchive, Result, SearchOptions, SearchWorkspace,
};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let pth = args
        .next()
        .expect("usage: retrieval <model.pth> <model.index>");
    let index = args
        .next()
        .expect("usage: retrieval <model.pth> <model.index>");

    let checkpoint = PthArchive::open(pth)?;
    let model = checkpoint.checkpoint().voice_model_info()?;
    let report = model.validate(checkpoint.checkpoint());
    assert!(report.is_valid(), "invalid model: {:?}", report.errors);

    let index = FaissIvfFlatIndex::open(index)?.load()?;
    model.validate_index_dimension(index.dimension())?;

    let mut workspace = index.workspace(8);
    let mut output = vec![0.0; index.dimension()];

    // A real application supplies one feature frame from its encoder.
    let query = index
        .reconstruct(0)
        .expect("example index has no ID 0")
        .to_vec();
    retrieve_frame(&index, &query, &mut output, &mut workspace)?;

    println!("retrieved {} neighbors", workspace.neighbors().len());
    Ok(())
}

fn retrieve_frame(
    index: &LoadedIvfFlatIndex,
    features: &[f32],
    output: &mut [f32],
    workspace: &mut SearchWorkspace,
) -> Result<()> {
    index.search_and_blend(
        features,
        output,
        SearchOptions { k: 8, nprobe: 1 },
        0.75,
        workspace,
    )?;
    Ok(())
}
