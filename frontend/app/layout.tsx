import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RoadLabelOps · CVAT Road Data Workflow",
  description: "Turn road video into reviewable CVAT tasks and traceable dataset releases.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
