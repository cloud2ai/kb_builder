#!/usr/bin/env python3
import argparse
import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List

from utils.vuepress import SidebarParser
from utils.markdown_splitter import MarkdownSplitter
from utils.kb_client import KBClient

# Paths for the knowledge base to keep track of the source and converted
# markdown files. During converted, if the markdown is not changed, the
# converted file will not be overwritten. That we can save the money for
# the LLM.
KB_BASE_PATH = 'kb'
KB_SRC_PATH = 'src'
KB_CONVERTED_PATH = 'converted'

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
            logger.debug(f"lang_code: {lang_code} ts_file_path: "
                         f"{ts_file_path}")

            # Use SidebarParser to parse the file
            vuepress_parser = SidebarParser(ts_file_path)
            sections = vuepress_parser.parse()
            if sections:
                language_map[lang_code] = sections
            else:
                logger.warning(f"No sections found for language: "
                               f"{lang_code}")

    logger.debug(f"language_map: {language_map}")
    return language_map

def process_markdown_files(menu_name: str, file_paths: List[str],
                           src_save_path: str, converted_save_path: str):
    """
    Process markdown files using LangChain text splitter and save the
    processed content to new files.
    """
    converted_markdown_paths = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue

        logger.debug(f"Processing file: {file_path}")
        #with open(file_path, 'r', encoding='utf-8') as f:
        #    content = f.read()

        markdown_splitter = MarkdownSplitter(markdown_path=file_path)
        converted_paths = markdown_splitter.split_and_save(
            src_save_path, converted_save_path, menu_name)
        converted_markdown_paths.extend(converted_paths)

    return converted_markdown_paths

def prepare_kb_dirs(vuepress_path: str):
    kb_src_path = os.path.join(vuepress_path, KB_BASE_PATH, KB_SRC_PATH)
    kb_converted_path = os.path.join(vuepress_path, KB_BASE_PATH,
                                    KB_CONVERTED_PATH)

    os.makedirs(kb_src_path, exist_ok=True)
    os.makedirs(kb_converted_path, exist_ok=True)
    logger.info(f"Created directory: {kb_src_path}")
    logger.info(f"Created directory: {kb_converted_path}")

    return kb_src_path, kb_converted_path

def prepare_kb_menu_dirs(kb_src_path: str, kb_converted_path: str,
                          menu_name: str):
    src_save_path = os.path.join(kb_src_path, menu_name)
    converted_save_path = os.path.join(kb_converted_path, menu_name)
    os.makedirs(src_save_path, exist_ok=True)
    os.makedirs(converted_save_path, exist_ok=True)
    logger.info(f"Created directory for {menu_name}: {src_save_path}")
    logger.info(f"Created directory for {menu_name}: {converted_save_path}")

    return src_save_path, converted_save_path

def check_and_upload_document(converted_path: Path, kb_client: KBClient,
                              lang: str) -> bool:
    """
    Check if document needs to be uploaded and handle the upload process.
    Returns True if upload was performed, False if skipped.

    Args:
        converted_path: Path to the converted markdown file
        kb_client: KBClient instance for uploading
    """
    uploaded_path = Path(str(converted_path) + '.uploaded')
    doc_name = str(converted_path).split('converted/')[-1]

    # Check if file was previously uploaded
    if uploaded_path.exists():
        with open(converted_path, 'r', encoding='utf-8') as f:
            current_content = f.read()
        with open(uploaded_path, 'r', encoding='utf-8') as f:
            uploaded_content = f.read()

        if current_content == uploaded_content:
            logger.info(f"Skipping {doc_name} - content unchanged")
            # Remove the unconverted file since we have an uploaded version
            converted_path.unlink()
            return False

    # Upload the document
    try:
        with open(converted_path, 'r', encoding='utf-8') as f:
            content = f.read()

        logger.info(f"Uploading document '{doc_name}' to knowledge base")
        doc_language = "English" if lang == "en" else "Chinese"
        kb_client.create_or_update_document(
            document_name=doc_name,
            text=content,
            doc_language=doc_language
        )
        logger.info(f"Successfully uploaded document: {doc_name}")

        # Save uploaded content
        with open(uploaded_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Remove the unconverted file
        converted_path.unlink()
        return True

    except Exception as e:
        logger.error(f"Failed to upload document {doc_name}: {e}")
        raise

def upload_files_to_kb(converted_markdown_paths: List[Path],
                       kb_client: KBClient, lang: str):
    """
    Process and upload converted markdown files to the knowledge base.
    """
    for converted_path in converted_markdown_paths:
        logger.debug(f"Processing converted file: {converted_path}")
        check_and_upload_document(Path(converted_path), kb_client, lang)

def main():
    args = parse_arguments()

    post_to_kb = args.kb_url is not None and args.kb_name is not None

    try:
        language_map = parse_sidebar_files(args.vuepress_path)
        kb_src_path, kb_converted_path = prepare_kb_dirs(args.vuepress_path)

        for lang, file_dict in language_map.items():
            logger.info(f"Processing files for language: {lang}")

            kb_client = None
            if post_to_kb:
                kb_client = KBClient(kb_name=f"{args.kb_name}_{lang}",
                                     base_url=args.kb_url)

            for menu_name, files in file_dict.items():
                absolute_paths = [
                    os.path.join(args.vuepress_path, "src", file.lstrip('/'))
                    for file in files
                ]
                logger.debug(f"menu_name: {menu_name}, absolute_paths: {absolute_paths}")

                src_save_path, converted_save_path = prepare_kb_menu_dirs(
                    kb_src_path, kb_converted_path, menu_name)

                converted_markdown_paths = process_markdown_files(
                    menu_name, absolute_paths,
                    src_save_path, converted_save_path)

                if post_to_kb:
                    upload_files_to_kb(
                        converted_markdown_paths, kb_client, lang)
    except Exception as e:
        logger.exception(f"Error: {str(e)}")
        return 1

    return 0

def parse_arguments():
    parser = argparse.ArgumentParser(description='Knowledge Base Builder '
                                     'for VuePress Documentation')
    parser.add_argument('--vuepress-path', required=True,
                        help='Path to VuePress documentation directory')
    parser.add_argument('--kb-url', required=False,
                        help='Knowledge base endpoint URL')
    parser.add_argument('--kb-name', required=False,
                        help='Knowledge base name')

    return parser.parse_args()

if __name__ == "__main__":
    exit(main())
