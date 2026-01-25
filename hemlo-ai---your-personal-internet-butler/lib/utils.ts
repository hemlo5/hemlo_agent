export function cn(...classes: Array<string | number | boolean | null | undefined>) {
  return classes
    .flatMap((c) => (typeof c === "string" ? c.split(" ") : c))
    .filter(Boolean)
    .join(" ");
}
