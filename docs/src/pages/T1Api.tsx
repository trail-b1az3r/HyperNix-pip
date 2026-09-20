import { ReferenceDocsPage } from './ReferenceDocs'

export function T1ApiPage() {
  return <ReferenceDocsPage
    dataUrl="./v1/t1-api.json"
    kicker="Reference · T1"
    title="T1 API"
    lede="The same expanded reference layout, focused on the T1 server, SDK, HTTP routes, required server modules, typed errors, and raw endpoint escape hatches."
    mode="t1"
  />
}
