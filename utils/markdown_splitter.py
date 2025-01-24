import re
import os
import logging
from pathlib import Path

from utils.llm_client import AzureOpenAIClient

# Initialize logger
logger = logging.getLogger(__name__)

DEFAULT_HEADING_LEVEL = 2

class MarkdownSplitter:

    def __init__(self, markdown_path: str):
        if markdown_path:
            try:
                with open(markdown_path, 'r', encoding='utf-8') as f:
                    self.content = f.read()
                logger.debug(f"Loaded markdown content from {markdown_path}")
            except Exception as e:
                logger.error(f"Error reading markdown file {markdown_path}: {e}")
                raise
        else:
            logger.error("markdown_path must be provided.")
            raise ValueError("markdown_path must be provided.")

        self.markdown_path = markdown_path
        self.llm_client = AzureOpenAIClient()

    def _is_file_changed(self, src_save_path: str, content: str) -> bool:
        """
        Check if the file has changed.
        """
        if os.path.exists(src_save_path):
            with open(src_save_path, 'r', encoding='utf-8') as file:
                existing_content = file.read()

            return not existing_content == content

        return True

    def split_and_save(self, src_save_path: str, converted_save_path: str,
                       menu_name: str, merge_content: bool = False):
        converted_paths = []
        sections = self.split_by_heading(menu_name)
        all_converted_content = []
        base_filename = Path(self.markdown_path).stem

        for index, section in enumerate(sections.items(), start=1):
            heading, content = section
            logger.debug(f"Section {index}: {heading} with content "
                         f"length {len(content)}")

            filename = f"{base_filename}-{index}.md"
            full_src_path = Path(src_save_path) / filename
            converted_filename = f"{base_filename}-{index}-converted.md"
            full_converted_path = Path(converted_save_path) / converted_filename

            if self._is_file_changed(full_src_path, content):
                logger.debug(f"Saving section to {full_src_path}")
                with open(full_src_path, 'w', encoding='utf-8') as section_file:
                    section_file.write(content)

                # Convert content and collect it
                converted_content = self._convert_content(content)
                all_converted_content.append(converted_content)

                # Save individual converted content
                with open(full_converted_path, 'w', encoding='utf-8') as conv_file:
                    conv_file.write(converted_content)
                converted_paths.append(full_converted_path)
            else:
                logger.debug(f"No changes in source file {full_src_path}")
                if full_converted_path.exists():
                    # If converted file exists, read and append its content
                    with open(full_converted_path, 'r', encoding='utf-8') as conv_file:
                        converted_content = conv_file.read()
                    all_converted_content.append(converted_content)
                    converted_paths.append(full_converted_path)
                else:
                    # Convert content if converted file doesn't exist
                    converted_content = self._convert_content(content)
                    all_converted_content.append(converted_content)
                    with open(full_converted_path, 'w', encoding='utf-8') as conv_file:
                        conv_file.write(converted_content)
                    converted_paths.append(full_converted_path)

        # Save all converted content to a single file only if merge_content is True
        if merge_content and all_converted_content:
            combined_filename = f"{base_filename}.md"
            full_converted_path = Path(converted_save_path) / combined_filename
            combined_content = '\n\n'.join(all_converted_content)

            with open(full_converted_path, 'w', encoding='utf-8') as converted_file:
                converted_file.write(combined_content)

            # Replace individual paths with combined file path if merging
            converted_paths = [full_converted_path]

        return converted_paths

    def split_by_heading(self, menu_name: str = None,
                         heading_level: int = DEFAULT_HEADING_LEVEL) -> dict[str, str]:
        """
        Split the markdown content into sections based on the specified heading
        level. Each section will contain the content under the corresponding
        heading.

        Args:
            heading_level (int): The level of headings to split by. Default is 2.

        Returns:
            List[str]: A list of sections split by the specified heading level.
        """
        sections = {}
        current_section = []
        heading_count = 0

        # Background content is the content before the first split heading
        background_content = []

        parent_heading_name = f"[NAV]{menu_name}"

        lines = self.content.splitlines()

        # Skip frontmatter if present
        # Frontmatter is the YAML metadata block at the beginning of markdown files
        # that starts and ends with '---'
        # Example frontmatter:
        # ---
        # title: HyperBDR Setup
        # icon: fa-solid fa-sliders
        # ---
        start_idx = 0
        if lines and lines[0].strip() == '---':
            for i, line in enumerate(lines[1:], 1):
                if line.strip() == '---':
                    start_idx = i + 1
                    break

        # TODO(Ray): I keep this code for optimize my article storage later,
        # maybe in a Graph database or others when I have a new idea.
        # Collect background content before first heading of specified level
        #for line in lines[start_idx:]:
        #    if line.startswith('#' * heading_level):
        #        break
        #    background_content.append(line)

        #background = '\n'.join(background_content)

        for line in lines[start_idx:]:
            # Skip [[toc]] lines
            if line.strip() == "[[toc]]":
                continue

            if line.startswith('#'):
                level = line.count('#')

                current_heading_name = line[level:].strip()
                # Check if the current heading level is less than the specified
                # heading level. If so, update the parent heading name to include
                # the current heading.
                if level < heading_level:
                    if parent_heading_name:
                        parent_heading_name = (
                            f"{parent_heading_name} -> {current_heading_name}"
                        )
                    else:
                        parent_heading_name = current_heading_name
                        logger.debug(
                            f"Parent heading name: {parent_heading_name}"
                        )

                # If the current heading level matches the specified heading
                # level, increment the heading count and add the parent heading
                # name to the section.
                if level == heading_level:
                    heading_count += 1
                    # Add parent heading name to the section
                    current_heading_name = (
                        f"{parent_heading_name} -> {current_heading_name}"
                    )
                    line = f"## {current_heading_name}"

                # If the current heading level matches the specified heading
                # level and the heading count is greater than 1, save the
                # section and start a new one.
                if level == heading_level and heading_count > 1:
                    # Save the section and start a new one
                    #sections.append('\n'.join(current_section))
                    sections[current_heading_name] = '\n'.join(current_section)
                    logger.debug(f"Added section: {current_heading_name}:"
                                 f" {current_section}")
                    current_section = []

            current_section.append(line)

        if current_section:
            sections[current_heading_name] = '\n'.join(current_section)
            logger.debug(f"Added final section: {current_heading_name}: "
                         f"{current_section}")

        return sections

    def _process_table(self, content: str) -> str:
        """
        Process the table in the markdown content.
        """
#        system_prompt = """
#You are a professional solutions engineer. You need to help me summary and convert all tables in markdown to plain sentences. Provide the output in plain text format, without any additional explanations or information.
#
#1. Please keep my navigator which start with [NAV].
#2. Summarize each row in the table by describing its function and purpose and keep the links if there are any.
#3. Please keep other parts of the content unchanged.
#4. Ensure the output language matches the original Markdown content.
#"""
        system_prompt = """
You are a professional solutions engineer. You need to help me summary and convert all tables in markdown to plain sentences. Provide the output in plain text format, without any additional explanations or information.

1. Please keep my navigator which start with [NAV].
2. Summarize each row in the table by describing its function and purpose and keep the links if there are any, please use the background outside table.
3. Please keep other parts of the content unchanged.
4. Ensure the output language matches the original Markdown content.
"""
        human_prompt = content

        response = self.llm_client.ask(system_prompt, human_prompt)
        return response

    def _convert_content(self, content: str) -> str:
        """
        Convert the content to a new format.
        """
        if self._contains_table(content):
            logger.debug(f"Table found in content: {content}")
            content = self._process_table(content)
        else:
            logger.debug(f"No table found in content: {content}")

        return content

    def _contains_table(self, markdown_text):
        """
        Check if the markdown content contains a table.
        """
        table_regex = r'((?:\| *[^|\r\n]+ *)+\|)'

        if re.search(table_regex, markdown_text, re.DOTALL):
            return True

        return False