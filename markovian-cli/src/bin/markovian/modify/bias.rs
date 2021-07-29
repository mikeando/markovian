use markovian_core::generator_wrapper::GeneratorWrapper;

pub fn bias(generator: GeneratorWrapper, power: f32) -> GeneratorWrapper {
    match generator {
        GeneratorWrapper::Bytes(gen) => {
            GeneratorWrapper::Bytes(gen.map_probabilities(|p| p.powf(power)))
        }
        GeneratorWrapper::String(gen) => {
            GeneratorWrapper::String(gen.map_probabilities(|p| p.powf(power)))
        }
    }
}
