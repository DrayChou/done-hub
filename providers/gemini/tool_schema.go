package gemini

// cleanSchemaRecursively visits schema positions, not arbitrary object values.
// In particular a property NAMED "const", "$ref" or "deprecated" is user data,
// not a schema keyword. Defaults/examples may also contain arbitrary objects.
func cleanSchemaRecursively(value interface{}) {
	schema, ok := value.(map[string]interface{})
	if !ok {
		return
	}

	for _, annotation := range []string{"$schema", "$comment", "examples", "deprecated"} {
		delete(schema, annotation)
	}

	// Pi's metadata is an unrestricted string-keyed object expressed as
	// patternProperties: {"^.*$": {}}. Omitting this redundant constraint preserves
	// its meaning. Do NOT drop restrictive patterns/value schemas: unsupported
	// constraints must fail upstream rather than silently permit different input.
	if patterns, ok := schema["patternProperties"].(map[string]interface{}); ok && len(patterns) == 1 {
		_, restrictsUnmatched := schema["additionalProperties"]
		_, tracksEvaluated := schema["unevaluatedProperties"]
		if anyValue, ok := patterns["^.*$"].(map[string]interface{}); ok && len(anyValue) == 0 && !restrictsUnmatched && !tracksEvaluated {
			delete(schema, "patternProperties")
		}
	}

	// Keep $ref/$defs/$id/$anchor, const, conditional constraints and additionalProperties.
	// Deleting these changes validation semantics. An unsupported upstream must
	// return a visible error; resolving general JSON Schema is outside this fix.
	for _, key := range []string{"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"} {
		if children, ok := schema[key].(map[string]interface{}); ok {
			for _, child := range children {
				cleanSchemaRecursively(child)
			}
		}
	}
	for _, key := range []string{"items", "additionalProperties", "contains", "not", "if", "then", "else", "propertyNames"} {
		cleanSchemaRecursively(schema[key])
	}
	for _, key := range []string{"anyOf", "oneOf", "allOf", "prefixItems"} {
		if branches, ok := schema[key].([]interface{}); ok {
			for _, branch := range branches {
				cleanSchemaRecursively(branch)
			}
		}
	}
}
