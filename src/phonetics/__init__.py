"""Mandarin phonetics public API with lazy optional ML dependencies."""
from importlib import import_module
_EXPORTS={"Encoder":(".model","SharedEncoder"),"create_loaders":(".data","create_loaders"),"load_experiment":(".experiment","load_experiment"),"prepare_experiment":(".experiment","prepare_experiment"),"validate_experiment":(".experiment","validate_experiment"),"default_experiment_config":(".experiment","default_experiment_config"),"make_generator":(".reproducibility","make_generator"),"seed_everything":(".reproducibility","seed_everything"),"train_experiment":(".training","train_experiment"),"assemble_dataset":(".preparation","assemble_dataset"),"prepare_acoustic_bundle":(".preparation","prepare_acoustic_bundle"),"validate_acoustic_bundle":(".preparation","validate_acoustic_bundle"),"prepare_corpus":(".preparation","prepare_corpus"),"validate_corpus_bundle":(".preparation","validate_corpus_bundle"),"prepare_dataset":(".preparation","prepare_dataset"),"validate_prepared":(".preparation","validate_prepared")}
_EXPORTS.update({"prepare_character_corpus": (".corpus", "prepare_character_corpus"),
                 "prepare_sound_inventory": (".sound_inventory", "prepare_sound_inventory"),
                 "prepare_acoustic_targets": (".acoustic", "prepare_acoustic_targets"),
                 "assemble_character_data": (".acoustic", "assemble_character_data"),
                 "select_seed": (".experiment", "select_seed")})
__all__=list(_EXPORTS)
def __getattr__(name):
    try:module,attribute=_EXPORTS[name]
    except KeyError:raise AttributeError(name) from None
    return getattr(import_module(module,__name__),attribute)
