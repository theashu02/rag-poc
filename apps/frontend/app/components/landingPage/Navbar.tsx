"use client";

import React, { useState } from "react";
import Link from "next/link";
import { Menu } from "lucide-react";
import { Logo } from "./Logo";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { useRouter } from "next/navigation";

const navItems = [
  { href: "#features", label: "Features" },
  { href: "#pricing", label: "Pricing" },
  { href: "#about", label: "About" },
  { href: "#blog", label: "Blog" },
  { href: "#contact", label: "Contact" },
];

export function Navbar() {
  const [open, setOpen] = useState(false);
  const router = useRouter();

  const handleClick = () => {
    router.push("/auth");
  }

  return (
    <header className="w-full sticky top-0 z-50 bg-background/80 backdrop-blur">
      <nav className="container max-w-7xl mx-auto flex items-center justify-between py-4 px-4 sm:px-6 lg:px-8">
        <Logo />

        {/* desktop navigation */}
        <ul className="hidden lg:flex gap-6 xl:gap-10">
          {navItems.map((item) => (
            <li key={item.href}>
              <Link
                href={item.href}
                className="text-muted-foreground hover:text-foreground transition-colors"
              >
                {item.label}
              </Link>
            </li>
          ))}
        </ul>

        {/* desktop CTAs */}
        <div className="hidden lg:flex gap-3">
          <Button size="lg" className="bg-amber-200 text-md rounded-sm cursor-pointer" onClick={handleClick}>
            SignIn/SignUp
          </Button>
        </div>

        {/* mobile menu */}
        <Sheet open={open} onOpenChange={setOpen}>
          <SheetTrigger asChild className="lg:hidden">
            <Button variant="ghost" size="icon" aria-label="Open menu">
              <Menu className="h-5 w-5" />
            </Button>
          </SheetTrigger>

          <SheetContent
            side="left"
            className="flex flex-col gap-6 pt-16 w-3/4 sm:w-1/2"
          >
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="text-lg font-medium"
                onClick={() => setOpen(false)}
              >
                {item.label}
              </Link>
            ))}

            <div className="mt-auto flex flex-col gap-3">
              <Button variant="secondary" onClick={() => setOpen(false)}>
                SignIn/SignUp
              </Button>
            </div>
          </SheetContent>
        </Sheet>
      </nav>
    </header>
  );
}