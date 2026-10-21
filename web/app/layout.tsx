import type { Metadata } from "next";
import Nav from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "Asteroid Hunter",
  description: "A machine-learning pipeline that finds moving asteroids in Zwicky Transient Facility images, shown step by step on real data.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Inter+Tight:wght@500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap"
        />
      </head>
      <body>
        <Nav />
        <main>{children}</main>
        <footer>
          <span>Asteroid Hunter · Machine Learning (702DB0C015) · MPSTME NMIMS · 2026-27 · Advik, Aaryan, Paras, Satvik</span>
          <span className="mono">Data: ZTF via IRSA · test night 2019-02-08 · model v4 · no discovery is claimed</span>
        </footer>
      </body>
    </html>
  );
}
