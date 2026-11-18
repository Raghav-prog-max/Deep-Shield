import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DEEP SHIELD | Media review",
  description: "Local, evidence-led image and video analysis for human review.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
