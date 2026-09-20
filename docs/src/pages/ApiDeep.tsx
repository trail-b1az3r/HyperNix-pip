import { ReferenceDocsPage } from './ReferenceDocs'

export function ApiDeepPage() {
  return <ReferenceDocsPage
    dataUrl="./v1/api-deep.json"
    kicker="Reference · expanded"
    title="In-depth API"
    lede="Source-generated reference for HyperNix: public symbols, real usage examples, detected runtime requirements, errors, deprecations, replacements, and Git-aware change history."
    mode="deep"
  />
}
