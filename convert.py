#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from q4nx import create_converter
from q4nx.model_assets import assemble_model_assets, get_default_flm_version


def convert_gguf_to_q4nx(gguf_path: str, q4nx_path: str, override_model_arch:str, weights_type: str = 'language', source_model: str = None, flm_version: str = None, deploy_tag: str = None, deploy_from: str = None, deploy_name: str = None):
    model = create_converter(gguf_path, override_model_arch)
    model.convert(q4nx_path=q4nx_path, weights_type=weights_type)
    if flm_version is None:
        flm_version = get_default_flm_version()
    assemble_model_assets(
        model.gguf_reader,
        model.q4nx_config,
        q4nx_path,
        source_model=source_model,
        flm_version=flm_version,
    )
    if deploy_tag:
        from q4nx.deploy import deploy_model
        deploy_model(
            q4nx_path,
            deploy_tag,
            model.model_arch,
            model_dir_name=deploy_name,
            deploy_from=deploy_from,
        )
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Convert GGUF model files to Q4NX format (output always named model.q4nx)',
        epilog='Examples:\n'
               '  python convert.py -i model.gguf\n'
               '  python convert.py -i model.gguf -o output_folder\n'
               '  python convert.py model.gguf output_folder\n'
               '  python convert.py model.gguf .\n'
               '  python convert.py -i vision_model.gguf -o output_folder -t vision',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add support for both flag-based and positional arguments
    parser.add_argument('input_file', nargs='?', help='Input GGUF file (positional)')
    parser.add_argument('output_folder', nargs='?', help='Output folder (positional, optional)')
    parser.add_argument('-i', '--input', dest='input_flag', help='Input GGUF file')
    parser.add_argument('-o', '--output', dest='output_flag', help='Output folder (optional, defaults to input file directory)')
    parser.add_argument('-t', '--type', dest='weights_type', default='language', help='Type of weights to convert (default: language)',
                        choices=['language', 'vision', 'audio'])
    parser.add_argument('-f', '--force', dest='force_model_type', default="", help="Model type. Empty string for automatic recognition from gguf file")
    parser.add_argument('-s', '--source-model', dest='source_model', default=None,
                        help="Source HF/ModelScope model for tokenizer/config assets. A local dir, an HF cache repo name, or a repo id like 'Qwen/Qwen3.5-9B'. If omitted, the GGUF's provenance metadata is followed (local HF cache first, then SDK download).")
    parser.add_argument('--flm-version', dest='flm_version', default=None,
                        help="flm_version to write into the generated config.json (default: detected from `flm --version`)" )
    parser.add_argument('-d', '--deploy', dest='deploy_tag', default=None, metavar='NAME:SIZE',
                        help="Deploy the converted model into flm's models directory and register it under this tag (e.g. 'qwen3.5-claude:9b'). Uses a user-level model_list.json (point FLM_CONFIG_PATH at it to make `flm run` see the tag).")
    parser.add_argument('--deploy-from', dest='deploy_from', default=None, metavar='SOURCE_TAG',
                        help="Official registry entry to copy defaults from (e.g. 'qwen3.5:9b'). Auto-detected from the model architecture if omitted.")
    parser.add_argument('--deploy-name', dest='deploy_name', default=None, metavar='DIR',
                        help="Directory name inside flm's models dir (default: derived from the deploy tag, e.g. Qwen3.5-Claude-9B-NPU2).")
    
    args = parser.parse_args()
    
    # Determine input file (prioritize flag, then positional)
    input_path = args.input_flag or args.input_file
    
    if not input_path:
        parser.error('Input file is required. Use -i <file> or provide as positional argument.')
    
    # Determine output folder (prioritize flag, then positional)
    output_folder = args.output_flag or args.output_folder
    
    # Check if input file exists
    if not os.path.exists(input_path):
        parser.error(f'Input file does not exist: {input_path}')
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_folder)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"[INFO] Converting {input_path} to {output_folder}...")
    convert_gguf_to_q4nx(input_path, output_folder, args.force_model_type, weights_type=args.weights_type, source_model=args.source_model, flm_version=args.flm_version, deploy_tag=args.deploy_tag, deploy_from=args.deploy_from, deploy_name=args.deploy_name)
    print(f"[INFO] Conversion complete! Output saved to {output_folder}")



if __name__ == "__main__":
    # for debug, give the path and ouptut path here by directly set the command line args
    import sys
    # sys.argv = ['convert.py', '-i', 'unsloth_gpt-oss-20b-Q4_0.gguf', '-o', 'unsloth-gotoss20b-q40']
            
    # sys.argv = ['convert.py', '-i', 'unsloth_gpt-oss-20b-Q4_1.gguf', '-o', 'unsloth-gotoss20b-q41']            
            
    # import sys
    # # sys.argv = ['convert.py', '-i', 'gemma-3-4b-it-Q4_1.gguf', '-o', 'unsloth-gemma3-q41']    
    # # main()


    # # sys.argv = ['convert.py', '-i', 'gemma3-mmproj-BF16.gguf', '-o', 'unsloth-gemma3-vision', '-t', 'vision']
    # # main()
    
    

    # # sys.argv = ['convert.py', '-i'
    # , 'medgemma3-mmproj-BF16.gguf', '-o', 'unsloth-medgemma3-vision', '-t', 'vision']
    # # main()
    # sys.argv = ['convert.py', '-i', 'Qwen3-VL-4B-Instruct-Q4_1.gguf', '-o', 'unsloth-qwen3vl-4b-q41' ]
    # main()             
    
    # sys.argv = ['convert.py', '-i', 'Qwen3-4B-Q4_1.gguf', '-o', 'unsloth-qwen3-4b-q41' ]
    # main()                 
    # sys.argv = ['convert.py', '-i', 'qwen3vl-4b-mmproj-BF16.gguf', '-o', 'unsloth-qwen3vl-vision', '-t', 'vision']
    # main()        
    
    
    
    #sys.argv = ['convert.py', '-i', 'qwen3_5vl-4b-mmproj-BF16.gguf', '-o', 'unsloth-qwen3_5vl-vision', '-t', 'vision']
     
    #sys.argv = ['convert.py', '-i', 'qwen3_5vl-9bmmproj-BF16.gguf', '-o', 'unsloth-qwen3_5_9bvl-vision', '-t', 'vision'] 
    
    
    #sys.argv = ['convert.py', '-i', 'Qwen3.5-4B-Q4_1.gguf', '-o', 'unsloth-qwen3_5_4bq41'] 
    
    #sys.argv = ['convert.py', '-i', 'Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf', '-o', 'unsloth-qwen3_59b_uncensored', "-f", "qwen3.5-9B"]     
    # sys.argv = ['convert.py', '-i', 'Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf', '-o', 'unsloth-qwen3_59b_uncensored', "-f", "qwen3.5-9B"]     
    
    # sys.argv = ['convert.py', '-i', 'Qwen3.5-9B-Q4_1.gguf', '-o', 'unsloth-qwen3_5_9bq41']     
    
    
    
    #sys.argv = ['convert.py', '-i', 'gemma-4-E2B-it-Q4_1.gguf', '-o', 'unsloth-gemma4-2b-it-q41']    
    
    #sys.argv = ['convert.py', '-i', 'gemma4-2b-mmproj.gguf', '-o', 'unsloth-gemma4-2b-vision', '-t', 'vision']     
    
    #sys.argv = ['convert.py', '-i', 'gemma4-2b-mmproj.gguf', '-o', 'unsloth-gemma4-2b-audio', '-t', 'audio']         
    # sys.argv = ['convert.py', '-i', 'debug_gemma4e2b_model.gguf', '-o', 'debug-gemma4-2b-audio', '-t', 'audio', '-f', 'gemma4']           
    main()
