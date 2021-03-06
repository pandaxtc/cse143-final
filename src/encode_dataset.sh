#!/bin/bash
#
# This is a rather minimal example Argbash potential
# Example taken from http://argbash.readthedocs.io/en/stable/example.html
#
# ARG_POSITIONAL_SINGLE([data_dir],["directory to read data from."])
# ARG_POSITIONAL_SINGLE([vocab_size],["size of vocabulary for sentencepiece."])
# ARG_POSITIONAL_SINGLE([model_type],["sentencepiece model type. unigram or bpe"])
# ARG_HELP([The general script's help msg])
# ARGBASH_GO()
# needed because of Argbash --> m4_ignore([
### START OF CODE GENERATED BY Argbash v2.9.0 one line above ###
# Argbash is a bash code generator used to get arguments parsing right.
# Argbash is FREE SOFTWARE, see https://argbash.io for more info
# Generated online by https://argbash.io/generate

set -e

die() {
    local _ret="${2:-1}"
    test "${_PRINT_HELP:-no}" = yes && print_help >&2
    echo "$1" >&2
    exit "${_ret}"
}

begins_with_short_option() {
    local first_option all_short_options='h'
    first_option="${1:0:1}"
    test "$all_short_options" = "${all_short_options/$first_option/}" && return 1 || return 0
}

# THE DEFAULTS INITIALIZATION - POSITIONALS
_positionals=()
# THE DEFAULTS INITIALIZATION - OPTIONALS

print_help() {
    printf '%s\n' "The general script's help msg"
    printf 'Usage: %s [-h|--help] <data_dir> <model_path> <vocab_size> <model_type>\n' "$0"
    printf '\t%s\n' "<data_dir>: \"directory to read data from.\""
    printf '\t%s\n' "<vocab_size>: \"size of vocabulary for sentencepiece.\""
    printf '\t%s\n' "<model_type>: \"sentencepiece model type. unigram or bpe\""
    printf '\t%s\n' "-h, --help: Prints help"
}

parse_commandline() {
    _positionals_count=0
    while test $# -gt 0; do
        _key="$1"
        case "$_key" in
        -h | --help)
            print_help
            exit 0
            ;;
        -h*)
            print_help
            exit 0
            ;;
        *)
            _last_positional="$1"
            _positionals+=("$_last_positional")
            _positionals_count=$((_positionals_count + 1))
            ;;
        esac
        shift
    done
}

handle_passed_args_count() {
    local _required_args_string="'data_dir', 'vocab_size' and 'model_type'"
    test "${_positionals_count}" -ge 3 || _PRINT_HELP=yes die "FATAL ERROR: Not enough positional arguments - we require exactly 4 (namely: $_required_args_string), but got only ${_positionals_count}." 1
    test "${_positionals_count}" -le 3 || _PRINT_HELP=yes die "FATAL ERROR: There were spurious positional arguments --- we expect exactly 4 (namely: $_required_args_string), but got ${_positionals_count} (the last one was: '${_last_positional}')." 1
}

assign_positional_args() {
    local _positional_name _shift_for=$1
    _positional_names="_arg_data_dir _arg_vocab_size _arg_model_type "

    shift "$_shift_for"
    for _positional_name in ${_positional_names}; do
        test $# -gt 0 || break
        eval "$_positional_name=\${1}" || die "Error during argument parsing, possibly an Argbash bug." 1
        shift
    done
}

parse_commandline "$@"
handle_passed_args_count
assign_positional_args 1 "${_positionals[@]}"

# OTHER STUFF GENERATED BY Argbash

### END OF CODE GENERATED BY Argbash (sortof) ### ])
# [ <-- needed because of Argbash

target_data_dir=${_arg_data_dir%/}_${_arg_model_type}_${_arg_vocab_size}/
model_prefix=models/sp_$(basename ${target_data_dir})

cp -r ${_arg_data_dir} ${target_data_dir}

spm_train \
    --input=<(cat ${target_data_dir}/train.src ${target_data_dir}/train.trg) \
    --model_prefix=${model_prefix} \
    --vocab_size=${_arg_vocab_size} \
    --model_type=${_arg_model_type}

for f in $(find ${target_data_dir} | grep -P './\w+.(src|trg)$'); do
    echo "$f > ${f%.*}.sp.${f##*.}"
    spm_encode --model=${model_prefix}.model $f >${f%.*}.sp.${f##*.}
done

cut -f1 ${model_prefix}.vocab > ${target_data_dir%/}/vocab.txt

# ] <-- needed because of Argbash
