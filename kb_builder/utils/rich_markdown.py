import re
import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from devtoolbox.llm.azure_openai_provider import AzureOpenAIConfig
from devtoolbox.llm.service import LLMService

# Initialize logger
logger = logging.getLogger(__name__)

DEFAULT_HEADING_LEVEL = 2

PROCESS_TABLE_PROMPT = """
You are a professional solutions engineer.

Your task: Convert all tables in the given Markdown content into natural
language descriptions, while preserving the surrounding non-table content
unchanged. Output the final result as plain text, without any additional
explanation.

Instructions:
- For every table in the Markdown:
  - Convert each row into a natural language sentence that clearly
    describes its function or purpose.
  - If a cell contains a link, keep the link in the sentence.
  - Use the background and context outside the table to guide how the
    information should be described.
- Do not change or summarize any content that is outside the tables.
  Keep it as-is.
- Ensure the output language remains the same as the input (e.g., if the
  Markdown is in Chinese, the output should also be in Chinese).
- Return the entire updated Markdown, replacing each table with the
  natural language version, and preserving everything else.
"""

SUMMARY_PROMPT = """
You are a professional technical writer.
Generate a concise, single-paragraph summary (max 200 characters) of a
product technical document.
Requirements:
- Preserve original meaning and technical context.
- Use key terms and phrases from the source text.
- Focus on core functionality, architecture, or usage details.
- Ensure technical accuracy and clarity.
- Optimize for RAG (Retrieval-Augmented Generation) systems.
Do not add explanations or formatting. Return only the summary.
"""

SUMMARY_QA_PROMPT = """
You are a professional technical writer.

Your task: Generate 10 insightful and relevant questions based on
the given technical document or paragraph.

Requirements:
- Questions must accurately reflect the content, context, and key
  concepts.
- Use terminology and phrasing from the original text whenever
  possible.
- Cover different aspects such as functionality, purpose, usage,
  limitations, architecture, etc.
- Ensure questions are clearly and precisely worded.
- Questions should be suitable for use in RAG (Retrieval-Augmented
  Generation), FAQs, or knowledge base indexing.

Instructions:
- Return only 10 questions, one per line.
- Do not include any additional formatting or explanation.
"""



@dataclass
class RichMarkdown:
    """
    Represents a rich markdown document with its structure and metadata.

    This class handles the parsing and processing of markdown documents,
    including caching, content conversion, and summary generation.
    """
    title: str = ""
    background: str = ""
    path: str = ""
    sections: List[Dict[str, str]] = field(default_factory=list)
    llm_service: LLMService = None
    total_word_count: int = field(init=False)
    document_summary: str = field(default="")
    metadata_loaded: bool = False
    metadata_path: str = field(init=False)
    menu_name: str = field(default="")

    def __post_init__(self):
        """
        Initialize document properties after initialization.
        """
        if not self.llm_service:
            raise ValueError("LLM service must be provided")

        # Calculate total word count
        self.total_word_count = (
            len(self.background.split()) +
            sum(len(section['content'].split()) for section in self.sections)
        )

        # Process content if not loaded from metadata
        if not self.metadata_loaded:
            self._process_content()

    def _process_content(self):
        """
        Process the content by converting tables and generating summaries.
        """
        # Convert tables in content
        self._convert_content()
        # Generate summaries
        self._generate_summaries()

    def _convert_content(self):
        """Convert tables in the content to plain text."""
        # Convert background content
        if self._contains_table(self.background):
            self.background = self._process_table(
                self.background
            )

        # Convert section contents
        for section in self.sections:
            if self._contains_table(section['content']):
                section['converted_content'] = self._process_table(
                    section['content']
                )

    def _generate_summaries(self):
        """Generate summaries for each section and the entire document."""
        # Generate section summaries
        section_summaries = []
        for section in self.sections:
            summary = self._summarize_text(
                section['content']
            )
            section['summary'] = summary
            section_summaries.append(
                f"Section '{section['title']}': {summary}"
            )

        # Generate background summary if exists
        if self.background.strip():
            background_summary = self._summarize_text(
                self.background
            )
            section_summaries.insert(0, f"Background: {background_summary}")

        # Generate document summary
        if section_summaries:
            combined_summaries = "\n".join(section_summaries)
            self.document_summary = self._summarize_text(
                combined_summaries,
                prompt=SUMMARY_PROMPT
            )

    def _summarize_text(
        self,
        text: str,
        prompt: str = None
    ) -> str:
        """
        Summarize text using LLM.

        Args:
            text: The text to summarize
            prompt: Optional custom prompt for summarization

        Returns:
            The summarized text
        """
        messages = [
            {
                "role": "system",
                "content": prompt if prompt else SUMMARY_QA_PROMPT
            },
            {
                "role": "user",
                "content": text
            }
        ]
        return self.llm_service.chat(messages)

    def _contains_table(self, markdown_text: str) -> bool:
        """Check if the markdown content contains a table."""
        table_regex = r'((?:\| *[^|\r\n]+ *)+\|)'
        return bool(re.search(table_regex, markdown_text, re.DOTALL))

    def _process_table(self, content: str) -> str:
        """Process the table in the markdown content."""
        messages = [
            {
                "role": "system",
                "content": PROCESS_TABLE_PROMPT
            },
            {
                "role": "user",
                "content": content
            }
        ]
        return self.llm_service.chat(messages)

    def _get_metadata_path(
        self,
        markdown_path: str,
        metadata_dir: Optional[str] = None
    ) -> str:
        """
        Get the metadata file path for a markdown file.

        Args:
            markdown_path: Path to the markdown file
            metadata_dir: Directory for metadata files

        Returns:
            Path to the metadata file
        """
        # Use the same filename as the converted file
        metadata_filename = f"{Path(markdown_path).stem}.metadata.json"
        metadata_path = os.path.join(metadata_dir, metadata_filename)
        logger.debug(f"Using metadata path: {metadata_path}")
        return metadata_path

    def _load_from_metadata(self, metadata_path: str) -> bool:
        """
        Load data from metadata file if it exists and is valid.

        Args:
            metadata_path: Path to the metadata file

        Returns:
            True if metadata was loaded successfully, False otherwise
        """
        if not os.path.exists(metadata_path):
            logger.debug(f"Metadata file not found: {metadata_path}")
            return False

        try:
            logger.debug(f"Attempting to load metadata from: {metadata_path}")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Verify metadata structure
            required_fields = [
                'title',
                'background',
                'sections'
            ]
            if not all(field in metadata for field in required_fields):
                missing_fields = [
                    f for f in required_fields
                    if f not in metadata
                ]
                logger.warning(
                    f"Metadata file {metadata_path} is missing fields: "
                    f"{missing_fields}"
                )
                return False

            # Load data from metadata
            self.title = metadata['title']
            self.background = metadata['background']
            self.sections = metadata['sections']
            self.total_word_count = metadata.get('total_word_count', 0)
            self.document_summary = metadata.get('document_summary', "")

            # Set metadata loaded flag
            self.metadata_loaded = True

            logger.info(
                f"Successfully loaded metadata from {metadata_path} "
                f"(sections: {len(self.sections)}, "
                f"word count: {self.total_word_count})"
            )
            return True
        except Exception as e:
            logger.error(
                f"Error loading metadata from {metadata_path}: {str(e)}"
            )
            return False

    def _save_to_metadata(self, metadata_path: str):
        """
        Save current state to metadata file.

        Args:
            metadata_path: Path to save metadata file
        """
        metadata = {
            'title': self.title,
            'background': self.background,
            'sections': self.sections,
            'total_word_count': self.total_word_count,
            'document_summary': self.document_summary,
            'menu_name': self.menu_name
        }

        try:
            # Ensure metadata directory exists
            metadata_dir = os.path.dirname(metadata_path)
            if metadata_dir:
                os.makedirs(metadata_dir, exist_ok=True)
                logger.debug(
                    f"Ensured metadata directory exists: {metadata_dir}"
                )

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.info(
                f"Successfully saved metadata to {metadata_path} "
                f"(sections: {len(self.sections)}, "
                f"word count: {self.total_word_count})"
            )
        except Exception as e:
            logger.error(
                f"Error saving metadata to {metadata_path}: {str(e)}"
            )

    def _load_and_validate_metadata(
        self,
        content: str,
        path: str,
        metadata_dir: Optional[str] = None,
        safe_filename: Optional[str] = None
    ) -> bool:
        """
        Load metadata for the markdown content.

        Args:
            content: The markdown content
            path: Path to the markdown file
            metadata_dir: Optional directory for metadata files
            safe_filename: Optional safe filename for metadata

        Returns:
            True if metadata was successfully loaded, False otherwise
        """
        # Get metadata path and check if file exists
        if safe_filename:
            metadata_path = os.path.join(
                metadata_dir,
                f"{safe_filename}.metadata.json"
            )
        else:
            metadata_path = self._get_metadata_path(path, metadata_dir)

        if not os.path.exists(metadata_path):
            logger.debug(f"No metadata file found at {metadata_path}")
            return False

        logger.debug(f"Found existing metadata file: {metadata_path}")

        # Try to load metadata
        if not self._load_from_metadata(metadata_path):
            logger.debug(f"Failed to load metadata from {metadata_path}")
            return False

        logger.info(
            f"Successfully loaded metadata for {path} "
            f"(sections: {len(self.sections)})"
        )
        return True

    def from_markdown(
        self,
        content: str,
        path: str,
        metadata_dir: Optional[str] = None,
        menu_name: str = "",
        safe_filename: Optional[str] = None
    ) -> 'RichMarkdown':
        """
        Parse markdown content into a rich markdown structure.

        The document must have:
        1. A single level 1 heading (#) as the document title
        2. Optional level 2 headings (##) for sections

        Content between level 1 heading and first level 2 heading
        (or end of file) will be treated as background information.

        Args:
            content: The markdown content to parse
            path: The path of the markdown file
            metadata_dir: Optional directory for metadata files
            menu_name: Name of the menu/section for navigation
            safe_filename: Optional safe filename for metadata

        Returns:
            self: The current instance with updated content

        Raises:
            ValueError: If document structure doesn't meet requirements
        """
        logger.info(f"Starting markdown parsing for file: {path}")
        logger.debug(f"Menu name: {menu_name}")

        # Initialize instance attributes
        self.path = path
        self.menu_name = menu_name

        # Set metadata path using safe_filename if provided
        if safe_filename:
            self.metadata_path = os.path.join(
                metadata_dir,
                f"{safe_filename}.metadata.json"
            )
        else:
            self.metadata_path = self._get_metadata_path(path, metadata_dir)

        logger.debug(f"Metadata path: {self.metadata_path}")

        # Try to load from metadata first
        if self._load_and_validate_metadata(
            content, path, metadata_dir, safe_filename
        ):
            logger.info(
                f"Successfully loaded metadata for {path} "
                f"(sections: {len(self.sections)})"
            )
            return self

        logger.debug("Metadata not found or invalid, starting fresh parsing")

        # Initialize parsing state
        document_title = None
        background_content = []
        sections = []
        current_section = {
            'title': None,
            'content': []
        }
        section_count = 0

        # Split content into lines
        lines = content.splitlines()
        logger.debug(f"Total lines in document: {len(lines)}")

        # Skip frontmatter if present
        # Frontmatter is the YAML metadata block at the beginning of markdown
        # files that starts and ends with '---'
        # Example frontmatter:
        # ---
        # title: HyperBDR Setup
        # icon: fa-solid fa-sliders
        # ---
        start_idx = 0
        if lines and lines[0].strip() == '---':
            logger.debug("Found frontmatter, skipping...")
            for i, line in enumerate(lines[1:], 1):
                if line.strip() == '---':
                    start_idx = i + 1
                    logger.debug(f"Frontmatter ends at line {start_idx}")
                    break

        in_code_block = False
        # Process each line
        for line_num, line in enumerate(lines[start_idx:], start_idx):
            # Skip table of contents marker
            if line.strip() == "[[toc]]":
                logger.debug(f"Found TOC marker at line {line_num}")
                continue

            # Check if we're inside a code block
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                logger.debug(
                    f"Code block {'started' if in_code_block else 'ended'} "
                    f"at line {line_num}"
                )

            # Process heading only outside code block
            if line.startswith('#') and not in_code_block:
                level = line.count('#')
                heading_text = line[level:].strip()
                logger.debug(
                    f"Found heading at line {line_num}: "
                    f"level {level}, text: {heading_text}"
                )

                if level == 1:
                    # Handle first heading level with #
                    # The first heading is the document title
                    if document_title is not None:
                        logging.warning(f"Current line is {line}")
                        logging.warning("Found a title but it's not the "
                                        "first heading, maybe it's inside "
                                        "code block")
                    else:
                        document_title = heading_text
                        logger.info(f"Document title: {document_title}")

                elif level == 2:
                    # Handle second heading level with ##
                    if current_section['title']:
                        section_content = '\n'.join(
                            current_section['content']
                        ).strip()
                        sections.append({
                            'title': current_section['title'],
                            'content': section_content,
                            'converted_content': section_content,
                            'summary': "",
                            'path': path,
                            'section_index': section_count
                        })
                        logger.debug(
                            f"Added section: {current_section['title']} "
                            f"(index: {section_count})"
                        )
                        section_count += 1

                    # Start new section
                    current_section = {
                        'title': heading_text,
                        'content': []
                    }
                    logger.debug(f"Starting new section: {heading_text}")
                else:
                    # Include lower heading levels in content
                    current_section['content'].append(line)
                    logger.debug(
                        f"Added subheading to section "
                        f"{current_section['title']}: {heading_text}"
                    )
            else:
                # Process non-heading content line
                if not current_section['title']:
                    background_content.append(line)
                    logger.debug(
                        f"Added line to background content: {line[:50]}..."
                    )
                else:
                    current_section['content'].append(line)
                    logger.debug(
                        f"Added line to section {current_section['title']}: "
                        f"{line[:50]}..."
                    )

        # Add final section if exists
        if current_section['title']:
            section_content = '\n'.join(
                current_section['content']
            ).strip()
            sections.append({
                'title': current_section['title'],
                'content': section_content,
                'converted_content': section_content,
                'summary': "",
                'path': path,
                'section_index': section_count
            })
            logger.debug(
                f"Added final section: {current_section['title']} "
                f"(index: {section_count})"
            )

        # Validate document structure
        if not document_title:
            error_msg = "Document must have a level 1 heading as title"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Update instance attributes
        self.title = document_title
        self.background = '\n'.join(background_content).strip()
        self.sections = sections

        logger.info(
            f"Successfully parsed document: {self.title} "
            f"(sections: {len(sections)}, "
            f"background length: {len(self.background)})"
        )

        # Process content and save metadata
        self.__post_init__()
        self._save_to_metadata(self.metadata_path)

        return self

    def get_section(
        self,
        index: int,
        converted: bool = True
    ) -> Dict[str, str]:
        """
        Get all information of a specific section by index.

        Args:
            index: The section index
            converted: Whether to return the converted content

        Returns:
            A dictionary containing section information:
            {
                'title': str,
                'content': str,
                'summary': str,
                'path': str,
                'section_index': int
            }

        Raises:
            IndexError: If section index is out of range
        """
        if 0 <= index < len(self.sections):
            section = self.sections[index].copy()
            if converted:
                section['content'] = section['converted_content']
            return section
        raise IndexError(f"Section index {index} out of range")

    def get_section_count(self) -> int:
        """
        Get the total number of sections.

        Returns:
            The number of sections in the document
        """
        return len(self.sections)

    def to_dict(self) -> Dict:
        """Convert the document structure to a dictionary."""
        return {
            'title': self.title,
            'background': self.background,
            'path': self.path,
            'total_word_count': self.total_word_count,
            'document_summary': self.document_summary,
            'section_count': len(self.sections),
            'sections': self.sections
        }

    def __str__(self) -> str:
        """Return a string representation of the document structure."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class MarkdownProcessor:
    """Processes markdown files and converts them into rich markdown structure.

    This class handles the parsing, processing, and conversion of markdown files
    into a structured format with rich metadata.
    """

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
        # Initialize LLM service
        openai_config = AzureOpenAIConfig(temperature=0.1, max_tokens=10000)
        self.llm_service = LLMService(openai_config)
        logger.debug("LLM service initialized with config: %s", openai_config)

    def generate_rich_markdown(
        self,
        src_save_path: str,
        converted_save_path: str,
        menu_name: str,
        safe_filename: str
    ):
        """
        Split markdown content into sections and save both original and converted versions.

        This method performs the following steps:
        1. Split the markdown content into sections based on headings
        2. Save each section as a separate file in the source directory
        3. Convert each section (e.g., process tables) and save to converted directory
        4. Optionally merge all converted content into a single file

        Args:
            src_save_path: Directory to save original split sections
            converted_save_path: Directory to save converted sections
            menu_name: Name of the menu/section for navigation
            safe_filename: Safe filename for the converted file
        """
        rich_markdown = RichMarkdown(
            llm_service=self.llm_service
        ).from_markdown(
            self.content,
            self.markdown_path,
            metadata_dir=converted_save_path,
            menu_name=menu_name,
            safe_filename=safe_filename
        )

        logger.debug(
            f"Split markdown into {rich_markdown.get_section_count()} sections"
        )
        logger.debug(
            "Document structure:\n"
            f"{rich_markdown}"
        )
        logger.debug(f"Metadata file path: {rich_markdown.metadata_path}")

        # Generate file paths using safe_filename
        converted_filename = f"{safe_filename}.txt"
        converted_path = Path(converted_save_path) / converted_filename
        metadata_filename = f"{safe_filename}.metadata.json"
        metadata_path = Path(converted_save_path) / metadata_filename

        merged_contents = []

        # Build top content before second level headings
        if rich_markdown.background:
            top_content = f"Background: {rich_markdown.background}"
            merged_contents.append(self._remove_empty_lines(top_content))

        # Add sections with their titles
        for i in range(rich_markdown.get_section_count()):
            # Get section information
            section = rich_markdown.get_section(i, converted=True)
            section_title = section['title']
            section_content = section['content']

            # Get summary if available
            summary = section['summary']
            logger.debug(
                f"Summary for section {section_title}: {summary}"
            )
            # FIXME(Ray): I removed title from the generate paragraph to
            # optimize fro RAG querying
            # merged_content = f"{section_title}\n"
            merged_content = ""

            # Add summary into final content if available
            if summary:
                merged_content += f"Summary for {section_title}: {summary}\n"
            merged_content += f"{section_content}\n"

            merged_contents.append(
                self._remove_empty_lines(merged_content)
            )

        merged_converted_content = "\n\n".join(merged_contents)

        logger.info(f"Writing converted content to {converted_path}")
        with open(converted_path, 'w', encoding='utf-8') as merged_file:
            merged_file.write(merged_converted_content)

        return str(converted_path), str(metadata_path)

    def _remove_empty_lines(self, content: str) -> str:
        """Remove empty lines from the content."""
        return '\n'.join(line for line in content.splitlines() if line.strip())