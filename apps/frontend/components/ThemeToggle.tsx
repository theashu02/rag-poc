"use client"

import * as React from "react"
import { Moon, Sun, LaptopMinimalCheck } from "lucide-react"
import { useTheme } from "next-themes"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

interface isCollapsed{
  isCollapsed: boolean;
}

export function ModeToggle({ isCollapsed }: isCollapsed) {
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = React.useState(false)

  React.useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) return null

  console.log(isCollapsed, "this comming from the sidebar")

  const renderThemeIcon = () => {
    switch (theme) {
      case "light":
        return <Sun className="h-[1.2rem] w-[1.2rem]" />
      case "dark":
        return <Moon className="h-[1.2rem] w-[1.2rem]" />
      case "system":
        return <LaptopMinimalCheck className="h-[1.2rem] w-[1.2rem]" />
      default:
        return <LaptopMinimalCheck className="h-[1.2rem] w-[1.2rem]" />
    }
  }

  const themeText =
    theme === "dark"
      ? "Theme: Dark"
      : theme === "light"
      ? "Theme: Light"
      : "Theme: System"

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" className="w-full text-sm text-black dark:text-amber-200 justify-center">
          {isCollapsed ? renderThemeIcon() : themeText}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={() => setTheme("light")}>
          <Sun className="mr-2 h-4 w-4" />
          <div>
            <p>Light</p>
            <p className="text-sm text-muted-foreground">Perfect for daytime browsing</p>
          </div>
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme("dark")}>
          <Moon className="mr-2 h-4 w-4" />
          <div>
            <p>Dark</p>
            <p className="text-sm text-muted-foreground">Easy on the eyes at night</p>
          </div>
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme("system")}>
          <LaptopMinimalCheck className="mr-2 h-4 w-4" />
          <div>
            <p>System</p>
            <p className="text-sm text-muted-foreground">Follows your device preference</p>
          </div>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}