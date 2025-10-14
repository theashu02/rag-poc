"use client";

import { Button } from "@/components/ui/button";
import { useSpacemanTheme } from "@space-man/react-theme-animation";
import { Moon, Sun } from "lucide-react";
import { useRef } from "react";

interface isCollapsed {
  isCollapsed: boolean;
}

export function ThemeToggle({ isCollapsed }: isCollapsed) {
  const { theme, switchThemeFromElement } = useSpacemanTheme();
  const buttonRef = useRef<HTMLButtonElement>(null);

  const handleToggle = async () => {
    const newTheme = theme === "light" ? "dark" : "light";
    if (buttonRef.current) {
      await switchThemeFromElement(newTheme, buttonRef.current);
    }
  };

  return (
    <>
      {isCollapsed ? (
        <Button
          ref={buttonRef}
          onClick={handleToggle}
          className="theme-toggle"
          aria-label="Toggle theme"
          variant="outline"
        >
          {theme === "light" ? <Moon size={20} /> : <Sun size={20} />}
        </Button>
      ) : (
        <>
          <Button
           ref={buttonRef}
           onClick={handleToggle}
            variant="outline"
            className="w-full text-sm text-black dark:text-amber-200 justify-center"
          >
            Theme: <span className="uppercase">{theme}</span>
          </Button>
        </>
      )}
    </>
  );
}
