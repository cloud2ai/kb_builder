#!/usr/bin/env python3
"""
Command line interface for Knowledge Base Builder.
This tool helps convert VuePress documentation into a knowledge base format.
"""

import argparse
import logging
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Set

from kb_builder.utils.vuepress import SidebarParser
from kb_builder.utils.rich_markdown import RichMarkdown
from kb_builder.utils.kb_client import KBClient

# Paths for the knowledge base to keep track of the source and converted
# markdown files. During converted, if the markdown is not changed, the
# converted file will not be overwritten. That we can save the money for
# the LLM.
KB_BASE_PATH = 'kb'

# Initialize logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_sidebar_files(vuepress_path: str) -> Dict[str, List[str]]:
    """
    Analyze files under src/.vuepress/sidebar to map languages to markdown
    files. Returns a dictionary mapping language codes to lists of
    markdown file paths.
    """
    sidebar_path = os.path.join(vuepress_path, "src", ".vuepress", "sidebar")
    language_map = {}

    if not os.path.exists(sidebar_path):
        raise ValueError(f"Sidebar directory not found at {sidebar_path}")

    for file in os.listdir(sidebar_path):
        if file.endswith('.ts'):
            lang_code = file.replace('.ts', '')
            ts_file_path = Path(os.path.join(sidebar_path, file))
            logger.debug(f"lang_code: {lang_code} ts_file_path: {ts_file_path}")

            # Use SidebarParser to parse the file
            vuepress_parser = SidebarParser(ts_file_path)
            sections = vuepress_parser.parse()
            if sections:
                language_map[lang_code] = sections
            else:
                logger.warning(f"No sections found for language: {lang_code}")

    logger.debug(f"language_map: {language_map}")
    return language_map


def get_safe_filename(file_path: str, menu_name: str) -> str:
    """
    Convert file path to a safe filename.
    - Convert to lowercase
    - Replace special characters with hyphens
    - Keep Chinese characters as is
    - Remove leading/trailing hyphens
    - Add menu name as prefix
    """
    # Get the relative path from src directory
    if 'src/' in file_path:
        file_path = file_path.split('src/')[-1]

    # Convert to lowercase and replace special characters
    safe_name = re.sub(r'[^a-z0-9\u4e00-\u9fff]', '-', file_path.lower())
    # Remove multiple consecutive hyphens
    safe_name = re.sub(r'-+', '-', safe_name)
    # Remove leading/trailing hyphens
    safe_name = safe_name.strip('-')

    # Add menu name as prefix if not already included
    menu_prefix = re.sub(r'[^a-z0-9\u4e00-\u9fff]', '-', menu_name.lower())
    menu_prefix = re.sub(r'-+', '-', menu_prefix).strip('-')

    # Only add menu prefix if it's not already at the start of the filename
    if not safe_name.startswith(menu_prefix):
        safe_name = f"{menu_prefix}-{safe_name}"

    return safe_name


def process_markdown_files(
    menu_name: str,
    file_paths: List[str],
    kb_path: str
):
    """
    Process markdown files using LangChain text splitter and save the
    processed content to new files.
    """
    converted_markdown_paths = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue

        logger.debug(f"Processing markdown file: {file_path}")
        # Read and process the markdown file
        #markdown_splitter = MarkdownProcessor(markdown_path=file_path)
        safe_filename = get_safe_filename(file_path, menu_name)
        #converted_path, metadata_path = markdown_splitter.generate_rich_markdown(
        #    kb_path, kb_path, menu_name, safe_filename)
        #converted_markdown_paths.append((converted_path, metadata_path))
        rich_markdown = RichMarkdown(
            markdown_path=file_path,
            converted_dir=kb_path,
            converted_filename=safe_filename,
            menu_name=menu_name
        )
        rich_markdown.generate_rich_markdown()
        rich_markdown.save_rich_markdown()
        converted_markdown_paths.append((
            rich_markdown.rich_markdown_path,
            rich_markdown.metadata_path
        ))

    return converted_markdown_paths


def prepare_kb_dirs(vuepress_path: str, lang: str = None) -> str:
    """
    Prepare the knowledge base directory structure.
    Creates a language-specific directory if lang is provided.

    Args:
        vuepress_path: Path to the VuePress documentation
        lang: Language code (optional)

    Returns:
        Path to the knowledge base directory
    """
    kb_path = os.path.join(vuepress_path, KB_BASE_PATH)
    if lang:
        kb_path = os.path.join(kb_path, lang)

    os.makedirs(kb_path, exist_ok=True)
    logger.info(f"Created directory: {kb_path}")
    return kb_path


def check_and_upload_document(
    converted_path: Path,
    metadata_path: Path,
    kb_client: KBClient,
    lang: str
) -> bool:
    """
    Check if document needs to be uploaded and handle the upload process.
    Returns True if upload was performed, False if skipped.

    Args:
        converted_path: Path to the converted markdown file
        metadata_path: Path to the metadata file
        kb_client: KBClient instance for uploading
        lang: Language code of the document
    """
    uploaded_path = Path(str(converted_path) + '.uploaded')
    doc_name = str(converted_path).split('converted/')[-1]

    # Check document status
    doc_exists = kb_client.check_document_exists(doc_name)
    if doc_exists:
        logger.debug(f"Remote document check for {doc_name}: exists")
    else:
        logger.debug(f"Remote document check for {doc_name}: not found")

    # Read current content
    with open(converted_path, 'r', encoding='utf-8') as f:
        current_content = f.read()

    # Check if content has changed
    content_unchanged = kb_client.check_content_unchanged(
        doc_name, current_content, uploaded_path
    )
    if content_unchanged:
        logger.debug(f"Local content check for {doc_name}: unchanged")
    else:
        logger.debug(f"Local content check for {doc_name}: changed")

    # Skip upload if document exists and content unchanged
    if doc_exists and content_unchanged:
        logger.info(
            f"Skipping {doc_name} - exists in KB and content unchanged"
        )
        converted_path.unlink()
        return False

    # Read metadata if exists
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            raw_metadata = json.load(f)
            supported_fields = {
                'title', 'background', 'document_summary',
                'menu_name', 'total_word_count'
            }
            metadata = {
                k: v for k, v in raw_metadata.items()
                if k in supported_fields
            }
        logger.info(f"Loaded metadata for {doc_name}")

    # Upload document
    logger.info(f"Uploading document '{doc_name}' to knowledge base")
    kb_client.upload_document(
        doc_name=doc_name,
        file_path=str(converted_path),
        lang=lang,
        metadata=metadata
    )

    # Save uploaded content
    kb_client.save_uploaded_content(current_content, uploaded_path)

    # Clean up
    converted_path.unlink()
    return True


def upload_files_to_kb(
    converted_markdown_paths: List[tuple],
    kb_client: KBClient,
    lang: str
):
    """
    Process and upload converted markdown files to the knowledge base.

    Args:
        converted_markdown_paths: List of tuples containing (converted_path, metadata_path)
        kb_client: KBClient instance for uploading
        lang: Language code of the document
    """
    for converted_path, metadata_path in converted_markdown_paths:
        logger.debug(f"Processing converted file: {converted_path}")
        check_and_upload_document(
            Path(converted_path),
            Path(metadata_path),
            kb_client,
            lang
        )


def get_all_processed_files(
    language_map: Dict[str, Dict[str, List[str]]],
    vuepress_path: str
) -> Dict[str, Set[str]]:
    """
    Get all processed file names for each language.
    Only include files that actually exist in the filesystem.

    Args:
        language_map: Dictionary mapping languages to menu files
        vuepress_path: Path to the VuePress documentation

    Returns:
        Dictionary mapping languages to sets of processed file names
    """
    processed_files = {}
    for lang, file_dict in language_map.items():
        lang_files = set()
        logger.info(f"Processing files for language: {lang}")
        for menu_name, files in file_dict.items():
            logger.info(f"Processing menu: {menu_name}")
            for file in files:
                absolute_path = os.path.join(
                    vuepress_path, "src", file.lstrip('/')
                )

                # Skip if file doesn't exist
                if not os.path.exists(absolute_path):
                    logger.warning(
                        f"File not found, skipping: {absolute_path}"
                    )
                    continue

                safe_name = get_safe_filename(absolute_path, menu_name)
                logger.debug(
                    f"Original file: {file} -> "
                    f"Safe name: {safe_name}"
                )
                lang_files.add(safe_name)

        processed_files[lang] = lang_files
        logger.info(
            f"Total processed files for {lang}: {len(lang_files)}"
        )
    return processed_files


def cleanup_cache_files(
    kb_path: str,
    processed_files: Set[str]
) -> None:
    """
    Clean up cache files that are not in the processed files list.
    Keep files if any processed file name is in the cache file name.

    Args:
        kb_path: Path to the knowledge base directory
        processed_files: Set of processed file names
    """
    logger.info("Starting cache cleanup process")
    logger.info(f"Processed files to keep: {sorted(processed_files)}")

    # Get all cache files
    cache_files = []
    for root, _, files in os.walk(kb_path):
        for file in files:
            if file.endswith(('.uploaded', '.metadata.json', '.txt')):
                cache_files.append(os.path.join(root, file))

    logger.info(f"Found {len(cache_files)} cache files in {kb_path}")

    # Find files to delete
    files_to_keep = []
    files_to_delete = []
    for cache_file in cache_files:
        file_name = os.path.basename(cache_file)
        file_base_name = file_name.split('.')[0]
        logger.debug(f"Checking cache file: {file_name}, "
                     f"base name: {file_base_name}")

        if file_base_name in processed_files:
            files_to_keep.append(cache_file)
            logger.debug(f"Keeping file: {cache_file}")
        else:
            files_to_delete.append(cache_file)
            logger.debug(f"Marked for deletion: {cache_file}")

    logger.info(
        f"Cache files summary:\n"
        f"- Total files: {len(cache_files)}\n"
        f"- Files to keep: {len(files_to_keep)}\n"
        f"- Files to delete: {len(files_to_delete)}"
    )

    # Delete files
    for cache_file in files_to_delete:
        logger.info(f"Deleting cache file: {cache_file}")
        try:
            os.remove(cache_file)
        except Exception as e:
            logger.error(
                f"Error deleting cache file {cache_file}: {str(e)}"
            )

    logger.info("Cache cleanup completed")


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Knowledge Base Builder for VuePress Documentation\n'
            'This tool helps you convert VuePress documentation into a '
            'knowledge base format.\n'
            'It processes markdown files and optionally uploads them to a '
            'knowledge base service.'
        )
    )
    parser.add_argument(
        '--vuepress-path',
        required=True,
        help=(
            'Path to your VuePress documentation directory. '
            'This should be the root directory containing the "src" folder.'
        )
    )
    parser.add_argument(
        '--lang',
        required=False,
        help=(
            'Language codes to process (e.g., "en" or "en,zh"). '
            'If not specified, all available languages will be processed.'
        )
    )
    parser.add_argument(
        '--kb-url',
        required=False,
        help=(
            'Knowledge base API endpoint URL. '
            'Example: http://dify.ai/v1. '
            'If not provided, the tool will only process files without '
            'uploading.'
        )
    )
    parser.add_argument(
        '--kb-name',
        required=False,
        help=(
            'Name of the knowledge base to create or update. '
            'This will be used as the prefix for language-specific knowledge '
            'bases. For example, if set to "docs", it will create "docs_en" '
            'and "docs_zh" for English and Chinese content.'
        )
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with detailed logging'
    )

    return parser.parse_args()


def setup_logging(debug: bool = False):
    """
    Configure logging based on debug mode.

    Args:
        debug: Whether to enable debug mode
    """
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    if debug:
        logger.debug("Debug mode enabled")

    return logger


def main():
    """
    Main entry point for the knowledge base builder.
    """
    args = parse_arguments()
    logger = setup_logging(args.debug)

    post_to_kb = args.kb_url is not None and args.kb_name is not None

    try:
        language_map = parse_sidebar_files(args.vuepress_path)
        logger.debug(f"Get language_map from sidebar files: {language_map}")

        # Filter languages if specified
        if args.lang:
            selected_langs = [lang.strip() for lang in args.lang.split(',')]
            language_map = {
                lang: files for lang, files in language_map.items()
                if lang in selected_langs
            }
            if not language_map:
                logger.warning(
                    f"No matching languages found for: {args.lang}. "
                    f"Available languages: {list(language_map.keys())}"
                )
                return 1

        # Get all processed files at the beginning
        logger.info("Collecting all files to be processed")
        processed_files = get_all_processed_files(
            language_map,
            args.vuepress_path
        )

        # Create base KB directory
        base_kb_path = prepare_kb_dirs(args.vuepress_path)
        logger.debug(f"Base KB Path: {base_kb_path}")

        for lang, file_dict in language_map.items():
            logger.info(f"Processing files for language: {lang}")

            # Create language-specific KB directory
            kb_path = prepare_kb_dirs(args.vuepress_path, lang)
            logger.debug(f"KB Path for {lang}: {kb_path}")

            kb_client = None
            if post_to_kb:
                kb_client = KBClient(
                    kb_name=f"{args.kb_name}_{lang}",
                    base_url=args.kb_url
                )
                kb_client.create_kb_metadata()

            for menu_name, files in file_dict.items():
                absolute_paths = [
                    os.path.join(args.vuepress_path, "src", file.lstrip('/'))
                    for file in files
                ]
                logger.debug(
                    f"menu_name: {menu_name}, "
                    f"absolute_paths: {absolute_paths}"
                )

                converted_markdown_paths = process_markdown_files(
                    menu_name,
                    absolute_paths,
                    kb_path
                )

                if not converted_markdown_paths:
                    logger.warning(
                        f"No converted markdown paths for {menu_name}"
                    )

                if post_to_kb:
                    upload_files_to_kb(
                        converted_markdown_paths,
                        kb_client,
                        lang
                    )

            # Clean up cache files for this language
            if post_to_kb:
                logger.info(f"Starting cache cleanup for language: {lang}")
                cleanup_cache_files(kb_path, processed_files[lang])

            # Clean up remote documents for this language
            if post_to_kb:
                logger.info(f"Starting remote document cleanup for {lang}")
                kb_client.cleanup_documents(list(processed_files[lang]))

    except Exception as e:
        logger.exception(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
