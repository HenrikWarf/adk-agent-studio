# Structured JSON Output Feature

## Overview

The agent template now supports **structured JSON output** through response schemas. This allows agents to return data in a consistent, parseable format instead of free-form text.

## What Was Added

### 1. Core Implementation (`agent_def/agent.py`)

- **`response_schema` support**: Added to `Agent` initialization
- **Automatic JSON parsing**: `send_message()` now automatically parses JSON responses when a schema is configured
- **`return_json` parameter**: Optional parameter to force JSON parsing even without a schema

### 2. Example Agent Configurations

Two example agents demonstrating structured output:

**`configs/product_analyzer_agent.json`**
- Analyzes product descriptions
- Returns structured data: name, category, features, price range, sentiment

**`configs/task_breakdown_agent.json`**
- Breaks down complex projects into tasks
- Returns structured data: project title, summary, tasks with priorities and dependencies

### 3. UI Enhancements (`app.py`)

- **Metadata Display**: Shows "✅ Enabled - Returns JSON" badge for agents with schemas
- **Schema Viewer**: Collapsible view of the response schema in agent details
- **JSON Display**: Pretty-printed JSON viewer for structured responses
- **Download Support**: Downloads as `.json` files with proper formatting
- **Chat Mode**: Handles both text and JSON responses in conversation view

### 4. Documentation

- **README.md**: Comprehensive "Structured Output" section with examples
- **example_structured_output.py**: Working examples demonstrating both agent types

## Usage

### In Configuration

```json
{
  "agent": {
    "response_schema": {
      "type": "object",
      "properties": {
        "field_name": {
          "type": "string",
          "description": "Field description"
        }
      },
      "required": ["field_name"]
    }
  }
}
```

### In Code

```python
# Automatic JSON parsing with response_schema
agent = AgentManager("configs/product_analyzer_agent.json")
await agent.initialize()
response = await agent.send_message("Analyze this product...")

# Response is a dict, not a string!
print(response["product_name"])

# Or explicitly request JSON
response = await agent.send_message("...", return_json=True)
```

### In UI

1. Select an agent with `response_schema`
2. See "✅ Enabled - Returns JSON" in metadata
3. View the schema in "📋 View Response Schema"
4. Responses display as formatted JSON with syntax highlighting
5. Download as `.json` files

## Benefits

✅ **Type Safety**: Define exact structure expected  
✅ **Validation**: Ensure required fields are present  
✅ **Easy Integration**: Direct use in applications  
✅ **Documentation**: Schema serves as API docs  
✅ **Consistency**: Same format every time  

## Schema Format

Follows JSON Schema specification:

- **type**: Data type (object, string, number, boolean, array)
- **properties**: Object field definitions
- **items**: Array element type
- **required**: Required field list
- **enum**: Allowed values
- **description**: Field documentation

## Best Practices

1. Use clear, descriptive field names
2. Add descriptions to help the AI understand expectations
3. Mark critical fields as required
4. Use enums for fields with limited options
5. Test with `gemini-2.0-flash-exp` or newer models for best results

## Testing

Run the example script:

```bash
python example_structured_output.py
```

This demonstrates:
- Product analysis with structured output
- Task breakdown with nested objects and arrays
- Accessing structured data programmatically

## Backward Compatibility

- ✅ Agents without `response_schema` continue to work as before (return strings)
- ✅ Existing configurations don't need modification
- ✅ `send_message()` maintains backward-compatible signature
- ✅ UI handles both string and JSON responses seamlessly

