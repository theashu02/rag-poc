"use client";

import React, { useState } from "react";
import Link from "next/link";
import { Menu, LogOut } from "lucide-react";
import { Logo } from "./Logo";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger, DropdownMenuSeparator, DropdownMenuLabel } from "@/components/ui/dropdown-menu";
import { useRouter } from "next/navigation";
import { useSession, signOut } from "next-auth/react";

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
  const { data: session } = useSession();

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
        <div className="hidden lg:flex gap-3 items-center">
          {session ? (
            <>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Avatar className="h-12 w-12 flex-shrink-0 cursor-pointer border-2 border-accent hover:shadow-lg transition rounded-full">
                    <AvatarImage
                      className="rounded-full"
                      src={session?.user.image || "/placeholder.svg"}
                      alt={session?.user.name || "Hi User"}
                    />
                    <AvatarFallback className="bg-sidebar-primary text-sidebar-primary-foreground">
                      {session?.user?.name?.[0] || "U"}
                    </AvatarFallback>
                  </Avatar>
                </DropdownMenuTrigger>
                <DropdownMenuContent
                  align="center"
                  className="w-56 rounded-lg shadow-lg border border-accent bg-popover mt-4"
                >
                  <DropdownMenuLabel className="flex flex-col items-start gap-1 px-4 py-3">
                    <span className="font-semibold text-sm">{session?.user?.name || "User"}</span>
                    <span className="block max-w-full truncate text-xs text-muted-foreground">{session?.user?.email || ""}</span>
                  </DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem
                    onClick={() => signOut()}
                    className="cursor-pointer px-4 py-2 text-sm rounded-md flex items-center gap-2 transition"
                  >
                    <LogOut className="w-4 h-4" />
                    Sign Out
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
              <span className="font-medium">{session?.user?.name?.split(" ")[0]}</span>
            </>
            ) : (
              <Button size="lg" className="bg-amber-200 text-md rounded-sm cursor-pointer" onClick={handleClick}>
                SignIn
              </Button>
            )}
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
            className="flex flex-col gap-6 pt-8 w-[90vw] max-w-xs h-full overflow-y-auto sm:w-1/2 sm:max-w-sm pl-8"
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

            <div className="mt-auto flex flex-col gap-3 mb-8">
              {session ? (
                <div className="flex items-center gap-2">
                  <Avatar className="h-10 w-10 flex-shrink-0">
                    <AvatarImage src={session?.user.image || "/placeholder.svg"} alt={session?.user.name || "Hi User"} />
                    <AvatarFallback className="bg-sidebar-primary text-sidebar-primary-foreground">
                      {session?.user?.name?.split(" ")[0]}
                    </AvatarFallback>
                  </Avatar>
                  <span className="font-medium">{session?.user?.name?.split(" ")[0]}</span>
                </div>
              ) : (
                <Button variant="secondary" onClick={() => setOpen(false)}>
                  SignIn/SignUp
                </Button>
              )}
            </div>
          </SheetContent>
        </Sheet>
      </nav>
    </header>
  );
}