"""
IPython Magic Commands for AI Assistant.

This module provides magic commands for using AI Assistant
directly in Databricks notebooks and Jupyter environments.

Features:
- %ai for single-line AI queries
- %%ai for multi-line prompts
- %ai_explain for code explanation
- %ai_optimize for SQL optimization
- %ai_generate for code generation
"""

from typing import Optional, Any
from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring


@magics_class
class AIMagics(Magics):
    """
    IPython magic commands for AI Assistant.

    These magics provide convenient shortcuts for AI operations
    in notebook environments.

    Usage:
        %load_ext ai_assistant.magic_commands
        %ai What is Delta Lake?
        %%ai_generate
        Create a function to read CSV files
    """

    def __init__(self, shell, assistant=None):
        super().__init__(shell)
        self._assistant = assistant
        self._last_response = None

    @property
    def assistant(self):
        """Get or create the AI assistant."""
        if self._assistant is None:
            # Try to get from user namespace
            self._assistant = self.shell.user_ns.get('ai_assistant')

            if self._assistant is None:
                # Try to create one
                try:
                    from . import AIAssistant
                    self._assistant = AIAssistant()
                    self.shell.user_ns['ai_assistant'] = self._assistant
                except Exception as e:
                    print(f"Could not create AI Assistant: {e}")
                    print("Please create one manually:")
                    print("  from ai_assistant import AIAssistant")
                    print("  ai_assistant = AIAssistant(secret_scope='your-scope')")
                    return None

        return self._assistant

    @line_magic
    @magic_arguments()
    @argument('prompt', nargs='*', help='The prompt to send to AI')
    @argument('-m', '--model', default=None, help='Model to use (gemini/claude)')
    @argument('-s', '--stream', action='store_true', help='Stream the response')
    def ai(self, line):
        """
        Send a prompt to the AI assistant.

        Usage:
            %ai What is Apache Spark?
            %ai -m gemini Explain Delta Lake
            %ai -s Stream this response

        Examples:
            %ai What are the best practices for Spark optimization?
            %ai -m claude How do I use Unity Catalog?
        """
        args = parse_argstring(self.ai, line)

        if not self.assistant:
            return

        prompt = ' '.join(args.prompt)
        if not prompt:
            print("Usage: %ai <your question>")
            return

        try:
            if args.stream:
                for chunk in self.assistant.stream(prompt, model=args.model):
                    print(chunk, end='', flush=True)
                print()
            else:
                response = self.assistant.ask(prompt, model=args.model)
                self._last_response = response
                print(response)

        except Exception as e:
            print(f"Error: {e}")

    @cell_magic
    @magic_arguments()
    @argument('-m', '--model', default=None, help='Model to use')
    @argument('-s', '--system', default=None, help='System instruction')
    def ai_cell(self, line, cell):
        """
        Send a multi-line prompt to the AI assistant.

        Usage:
            %%ai_cell
            Your multi-line
            prompt here

            %%ai_cell -m claude -s "You are a SQL expert"
            Optimize this query:
            SELECT * FROM table WHERE ...
        """
        args = parse_argstring(self.ai_cell, line)

        if not self.assistant:
            return

        try:
            response = self.assistant.ask(
                cell,
                model=args.model,
                system_instruction=args.system
            )
            self._last_response = response
            print(response)

        except Exception as e:
            print(f"Error: {e}")

    @line_magic
    @magic_arguments()
    @argument('-v', '--var', default=None, help='Variable containing code')
    @argument('-d', '--detail', default='medium',
              choices=['brief', 'medium', 'detailed'],
              help='Level of detail')
    def ai_explain(self, line):
        """
        Explain code using AI.

        Usage:
            %ai_explain -v my_code_variable
            %ai_explain -v sql_query -d detailed

        The variable should contain the code to explain.
        """
        args = parse_argstring(self.ai_explain, line)

        if not self.assistant:
            return

        if not args.var:
            print("Usage: %ai_explain -v <variable_name>")
            return

        code = self.shell.user_ns.get(args.var)
        if code is None:
            print(f"Variable '{args.var}' not found")
            return

        detail_prompts = {
            'brief': "Briefly explain (2-3 sentences) what this code does:",
            'medium': "Explain what this code does and how it works:",
            'detailed': "Provide a detailed line-by-line explanation of this code:"
        }

        prompt = f"""{detail_prompts[args.detail]}

```
{code}
```"""

        try:
            response = self.assistant.ask(
                prompt,
                system_instruction="You are a code explanation expert. Be clear and concise."
            )
            self._last_response = response
            print(response)

        except Exception as e:
            print(f"Error: {e}")

    @cell_magic
    @magic_arguments()
    @argument('-m', '--model', default=None, help='Model to use')
    def ai_optimize(self, line, cell):
        """
        Optimize a SQL query using AI.

        Usage:
            %%ai_optimize
            SELECT *
            FROM my_table
            WHERE column = 'value'
        """
        args = parse_argstring(self.ai_optimize, line)

        if not self.assistant:
            return

        prompt = f"""Optimize this Spark SQL query:

```sql
{cell}
```

Provide:
1. Issues with the current query
2. Optimized version
3. Explanation of changes"""

        try:
            response = self.assistant.ask(
                prompt,
                model=args.model,
                system_instruction="You are a Spark SQL optimization expert."
            )
            self._last_response = response
            print(response)

        except Exception as e:
            print(f"Error: {e}")

    @cell_magic
    @magic_arguments()
    @argument('-l', '--language', default='python', help='Programming language')
    @argument('-t', '--tests', action='store_true', help='Include unit tests')
    @argument('-o', '--output', default=None, help='Variable to store result')
    def ai_generate(self, line, cell):
        """
        Generate code using AI.

        Usage:
            %%ai_generate
            Create a function to calculate factorial

            %%ai_generate -l sql -o result
            Create a query to find top customers

            %%ai_generate -t
            Create a function to validate email addresses
        """
        args = parse_argstring(self.ai_generate, line)

        if not self.assistant:
            return

        prompt = f"""Generate {args.language} code for:

{cell}

Requirements:
- Write clean, production-ready code
- Include necessary imports
- Add comments explaining the code
{f'- Include comprehensive unit tests' if args.tests else ''}

Provide only the code, no additional explanations."""

        try:
            response = self.assistant.ask(
                prompt,
                system_instruction=f"You are an expert {args.language} programmer."
            )
            self._last_response = response

            # Store in variable if requested
            if args.output:
                self.shell.user_ns[args.output] = response
                print(f"Code stored in variable: {args.output}")

            print(response)

        except Exception as e:
            print(f"Error: {e}")

    @line_magic
    def ai_last(self, line):
        """
        Get the last AI response.

        Usage:
            %ai_last
            result = %ai_last
        """
        if self._last_response:
            return self._last_response
        else:
            print("No previous response")
            return None

    @line_magic
    @magic_arguments()
    @argument('-v', '--var', required=True, help='Variable containing error')
    @argument('-c', '--code', default=None, help='Variable containing code')
    def ai_fix(self, line):
        """
        Get help fixing an error.

        Usage:
            %ai_fix -v error_message
            %ai_fix -v error_message -c my_code
        """
        args = parse_argstring(self.ai_fix, line)

        if not self.assistant:
            return

        error = self.shell.user_ns.get(args.var)
        if error is None:
            print(f"Variable '{args.var}' not found")
            return

        code = None
        if args.code:
            code = self.shell.user_ns.get(args.code)

        try:
            response = self.assistant.explain_error(
                str(error),
                code=code
            )
            self._last_response = response
            print(response)

        except Exception as e:
            print(f"Error: {e}")

    @line_magic
    def ai_help(self, line):
        """
        Show help for AI magic commands.

        Usage:
            %ai_help
        """
        help_text = """
AI Assistant Magic Commands
===========================

Available Commands:
-------------------

%ai <prompt>
    Send a quick question to the AI.
    Options: -m (model), -s (stream)
    Example: %ai What is Delta Lake?

%%ai_cell
    Send a multi-line prompt.
    Options: -m (model), -s (system instruction)

%ai_explain -v <variable>
    Explain code stored in a variable.
    Options: -d (detail level: brief/medium/detailed)

%%ai_optimize
    Optimize a SQL query (write query in cell).

%%ai_generate
    Generate code from description.
    Options: -l (language), -t (include tests), -o (output variable)

%ai_fix -v <error_var> [-c <code_var>]
    Get help fixing an error.

%ai_last
    Get the last AI response.

%ai_help
    Show this help message.

Setup:
------
1. Load the extension:
   %load_ext ai_assistant.magic_commands

2. Or create assistant manually:
   from ai_assistant import AIAssistant
   ai_assistant = AIAssistant(secret_scope="your-scope")
"""
        print(help_text)


def load_ipython_extension(ipython):
    """
    Load the AI Assistant magic commands extension.

    Usage in notebook:
        %load_ext ai_assistant.magic_commands
    """
    # Check if assistant exists in namespace
    assistant = ipython.user_ns.get('ai_assistant')

    # Register magics
    magics = AIMagics(ipython, assistant)
    ipython.register_magics(magics)

    print("AI Assistant magic commands loaded!")
    print("Type %ai_help for usage information.")


def unload_ipython_extension(ipython):
    """Unload the extension."""
    pass
