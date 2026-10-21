"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  ["/", "Overview"],
  ["/pipeline/", "How it works"],
  ["/model/", "The model"],
  ["/results/", "Results"],
  ["/team/", "Team"],
] as const;

export default function Nav() {
  const path = usePathname();
  const here = (href: string) => (href === "/" ? path === "/" : path.startsWith(href.replace(/\/$/, "")));
  return (
    <div className="fnav-wrap">
      <div className="fnav-bar">
        <Link href="/" className="brand" aria-label="Asteroid Hunter home">
          <i /> <span>Asteroid Hunter</span>
        </Link>
        <nav className="fnav" aria-label="Main">
          {LINKS.map(([href, label]) => (
            <Link key={href} href={href} className={here(href) ? "on" : ""}>
              {label}
            </Link>
          ))}
        </nav>
      </div>
    </div>
  );
}
