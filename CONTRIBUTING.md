# Contributing to VuePress2Dify

Thank you for your interest in contributing to VuePress2Dify! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## Contributor License Agreement (CLA)

Before contributing to this project, you must agree to the following terms:

### 1. Copyright License

You grant OneProCloud a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare derivative works of, publicly display, publicly perform, sublicense, and distribute your contributions.

### 2. Patent License

You grant OneProCloud a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer your contributions.

### 3. Representations

You represent that:
- You are legally entitled to grant the above licenses
- Your contributions are your original work
- Your contributions do not violate any third party's rights
- Your contributions do not contain any malicious code

## How to Contribute

### 1. Fork and Clone

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/vuepress2dify.git
   ```

### 2. Set Up Development Environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

### 3. Make Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes
3. Write or update tests
4. Update documentation

### 4. Code Style

- Follow PEP 8 style guide
- Use type hints
- Write docstrings for all functions and classes
- Keep lines under 100 characters
- Use meaningful variable names

### 5. Testing

- Write tests for new features
- Ensure all tests pass:
  ```bash
  pytest
  ```
- Maintain or improve test coverage

### 6. Documentation

- Update README.md if needed
- Add or update docstrings
- Update any relevant documentation
- Include examples for new features

### 7. Commit Changes

1. Stage your changes:
   ```bash
   git add .
   ```

2. Commit with a descriptive message:
   ```bash
   git commit -m "Description of changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

### 8. Pull Request

1. Create a pull request from your fork to the main repository
2. Fill out the pull request template
3. Wait for review and address any feedback

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the documentation with any new features or changes
3. The PR will be merged once it has been reviewed and approved

## Development Guidelines

### Code Structure

- Keep the code modular and maintainable
- Follow the existing project structure
- Use appropriate design patterns
- Write clean, readable code

### Error Handling

- Use appropriate exception handling
- Provide meaningful error messages
- Log errors appropriately

### Performance

- Consider performance implications
- Optimize where necessary
- Document any performance considerations

### Security

- Follow security best practices
- Handle sensitive data appropriately
- Validate all inputs
- Use secure coding practices

## Getting Help

If you need help or have questions:

1. Check the documentation
2. Search existing issues
3. Create a new issue if needed

## License

By contributing to this project, you agree that your contributions will be licensed under the project's Apache License 2.0.