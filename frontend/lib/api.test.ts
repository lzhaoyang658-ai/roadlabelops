import { describe, expect, it } from "vitest";
import { cvatJobUrl, cvatProjectUrl, thumbnailUrl } from "@/lib/api";

describe("CVAT links", () => {
  it("builds direct project and job URLs", () => {
    expect(cvatProjectUrl(901)).toBe("http://localhost:8080/projects/901");
    expect(cvatJobUrl(902, 903)).toBe("http://localhost:8080/tasks/902/jobs/903");
  });

  it("encodes scene identifiers in thumbnail URLs", () => {
    expect(thumbnailUrl("session one/scene 01")).toBe(
      "/api/v1/scenes/session%20one%2Fscene%2001/thumbnail",
    );
  });
});
