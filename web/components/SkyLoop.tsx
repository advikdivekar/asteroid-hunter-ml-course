"use client";

import { useEffect, useState } from "react";

/** The hero picture: the five real difference images of the test night, played as a loop, so the asteroid visibly moves. */
export default function SkyLoop() {
  const [frame, setFrame] = useState(1);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const id = setInterval(() => setFrame((f) => (f % 5) + 1), 700);
    return () => clearInterval(id);
  }, []);
  return (
    <div className="hero-art" aria-label="Five photos of the same patch of sky; one dot moves: the known asteroid 1996 XG5">
      {[1, 2, 3, 4, 5].map((k) => (
        <img key={k} src={`/data/scene/f${k}_science.png`} alt="" style={{ position: "absolute", inset: 0, opacity: k === frame ? 1 : 0 }} />
      ))}
      <div className="tag-on row">
        <span className="chip fill">photo {frame} of 5</span>
        <span className="chip" style={{ color: "#fff", background: "rgba(0,0,0,.5)" }}>
          ZTF · 2019-02-08 · real pixels
        </span>
      </div>
    </div>
  );
}
