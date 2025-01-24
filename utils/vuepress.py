"""
utils/vuepress.py

This module contains the SidebarParser class, which is responsible for
parsing VuePress sidebar configuration files. It extracts the hierarchical
structure of markdown files from the sidebar configuration and converts
JavaScript-like objects to valid JSON strings for further processing.

Dependencies:
- re: For regular expression operations used in parsing.
"""

import re
import json
import logging
from typing import Dict, List, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class SidebarParser:
    """Parser for VuePress sidebar configuration files.

    Extracts hierarchical structure of markdown files from sidebar config.
    """

    def __init__(self, content: Union[str, Path] = None):
        """Initialize SidebarParser.

        Args:
            content: Sidebar content string or Path to sidebar file
        """
        self.content = None
        if content:
            if isinstance(content, Path):
                logger.info(f"Reading sidebar file from: {content}")
                with open(content, 'r', encoding='utf-8') as f:
                    self.content = f.read()
                    logger.debug(f"Loaded content: {self.content[:200]}...")
            else:
                self.content = content
                logger.debug(f"Using provided content: {content[:200]}...")

    def _convert_to_valid_json(self, js_code: str) -> str:
        """Convert JavaScript object to valid JSON string.

        Args:
            js_code: JavaScript object code as string

        Returns:
            Valid JSON string
        """
        logger.debug(f"Converting JS code to JSON: {js_code[:200]}...")

        # Remove single-line comments
        js_code = re.sub(r'//.*', '', js_code)
        logger.debug("Removed single-line comments")

        # Ensure property names are double-quoted
        js_code = re.sub(r'(\w+):', r'"\1":', js_code)
        logger.debug("Added quotes to property names")

        # Remove trailing whitespace
        js_code = js_code.strip()

        # Remove content after last closing brace
        while not js_code.rstrip().endswith('}'):
            js_code = js_code[:-1]
        logger.debug("Removed content after last brace")

        # Remove trailing commas
        js_code = re.sub(r',\s*}', '}', js_code)
        js_code = re.sub(r',\s*]', ']', js_code)
        logger.debug("Removed trailing commas")

        # Remove unnecessary newlines in JSON
        js_code = re.sub(r'\s*\n\s*', '', js_code)

        # Replace single quotes with double quotes
        js_code = js_code.replace("'", '"')

        logger.debug(f"Final JSON: {js_code[:200]}...")
        return js_code

    def _extract_sidebar_config(self, content: str) -> Dict:
        """Extract sidebar configuration from TypeScript/JavaScript content.

        Args:
            content: File content as string

        Returns:
            Parsed sidebar configuration as dict
        """
        logger.info("Extracting sidebar configuration")
        # Split by 'sidebar(' to find sidebar configuration
        parts = content.split('sidebar(')
        logger.debug(f"Found {len(parts)} parts after splitting by 'sidebar('")

        # Look for sidebar configuration in parts
        for part in parts[1:]:
            if part.strip().startswith("{") and part.strip().endswith(");"):
                logger.debug("Found sidebar configuration block")
                valid_json = self._convert_to_valid_json(part)
                try:
                    config = json.loads(valid_json)
                    logger.info("Successfully parsed sidebar JSON")
                    return config
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse sidebar JSON")
                    raise ValueError(f"Failed to parse sidebar JSON: {str(e)}")

        logger.warning("No valid sidebar configuration found in file")
        return {}

    def parse_menu_items(self, root_path: str, menu_items: List[Dict],
                         parent_key: str = "") -> Dict[str, List[str]]:
        """Parse menu items and return markdown files for each path."""
        result = {}
        logger.info(f"Parsing menu items for root path: {root_path} "
                    f"with parent key: {parent_key}")

        for item in menu_items:
            current_text = item.get('text', '')
            current_key = current_text
            if parent_key:
                current_key = f"{parent_key}|{current_text}"
            logger.debug(f"Processing item: {current_text} "
                         f"with key: {current_key}")

            prefix = item.get('prefix', '')

            if 'children' not in item:
                logger.debug(f"Item '{current_text}' has no children.")
                continue

            # Skip if children is a string
            if isinstance(item['children'], str):
                logger.warning(f"Skipping item '{current_text}' as children is a string.")
                continue

            if isinstance(item['children'], list):
                # Check for nested structure in children
                # A nested structure looks like:
                # children: [
                #   {
                #     text: "Some text",
                #     children: [...]  # Nested items here
                #   },
                #   {
                #     text: "Other text",
                #     children: [...]  # More nested items
                #   }
                # ]
                has_nested = False
                for child in item['children']:
                    if isinstance(child, dict) and 'children' in child:
                        has_nested = True
                        logger.debug(f"Item '{current_text}' has nested children.")
                        break

                # Check if the current item has nested children
                # If the item has a nested structure, it will look like this:
                # {
                #   text: "Parent",
                #   children: [
                #     { text: "Child 1", children: [...] },
                #     { text: "Child 2", children: [...] }
                #   ]
                # }
                if has_nested:
                    # Handle nested menu items by recursively parsing them
                    logger.info(f"Recursively parsing nested items "
                                f"under '{current_text}'.")
                    nested_results = self.parse_menu_items(
                        root_path, item['children'], current_key)
                    result.update(nested_results)
                else:
                    # Handle flat menu items (items without nested structure)
                    logger.info(f"Processing flat menu items "
                                f"under '{current_text}'.")
                    if current_key not in result:
                        result[current_key] = []

                    markdown_files = self._process_flat_children(
                        root_path, current_key, item, prefix, result)
                    result[current_key].extend(markdown_files)

        return result

    def _process_flat_children(self, root_path: str, current_key: str,
                               item: Dict, prefix: str,
                               result: Dict[str, List[str]]) -> List[str]:
        """
        Process each child in the current item's children to generate a list
        of markdown file paths. If a child is a string, it is treated as a
        markdown file. If a child is a dictionary with a 'link', it is
        treated as a link to a markdown file.

        Args:
            root_path (str): The root path to prepend to markdown file paths.
            current_key (str): The current key in the result dictionary.
            item (Dict): The current item containing children to process.
            prefix (str): An optional prefix to prepend to child names.
            result (Dict[str, List[str]]): The result dictionary to update.

        Returns:
            List[str]: A list of processed markdown file paths.
        """

        markdown_files = []

        for child in item['children']:
            # Check if the child is a string.
            if isinstance(child, str):
                # If the child is a string, add the prefix if it exists.
                if prefix:
                    child = f"{prefix}{child}"

                # Ensure the child ends with '.md' for markdown files.
                if not child.endswith('.md'):
                    child = f"{child}.md"

                # Prepend root_path to the child.
                child = f"{root_path}/{child}"
                markdown_files.append(child)

            # Check if the child is a dictionary.
            elif isinstance(child, dict) and 'link' in child:
                # Handle dictionary items in the flat list.
                # Example content: { text: "Example", link: "example-link" }
                link = child['link']

                if link:
                    # Ensure the link ends with '.md'.
                    if not link.endswith('.md'):
                        link = f"{link}.md"

                    # Prepend root_path to the link.
                    link = f"{root_path}/{link}"
                    markdown_files.append(link)

        return markdown_files


    def parse(self) -> Dict[str, List[str]]:
        """Parse VuePress sidebar content into section->files mapping.

        Returns:
            Dict mapping section paths to lists of markdown files
        """
        if not self.content:
            logger.warning("No content provided to parse")
            return {}

        logger.info("Starting sidebar parsing")
        # Parse sidebar configuration
        config = self._extract_sidebar_config(self.content)

        if not config:
            logger.warning("No valid sidebar configuration found in file")
            return {}

        # Extract markdown files for each section
        final_results = {}
        for root_path, menu_items in config.items():
            logger.debug(f"Processing section: {root_path} {menu_items}")
            results = self.parse_menu_items(root_path, menu_items)
            logger.debug(f"Results: {results}")
            if results:
                final_results.update(results)
            else:
                logger.warning(f"No results found for section: {root_path}")

        logger.debug("="*100)
        logger.debug(f"Final results: {final_results}")
        logger.debug("="*100)

        return final_results
