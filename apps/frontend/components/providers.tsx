'use client'

import { ReactNode, Suspense } from "react"
import { ThemeProvider } from "@/components/theme-provider"
import SessionWrapping from "@/lib/sessionWrapping"
import { Toaster } from "@/components/ui/sonner"

export function Providers({ children }: { children: ReactNode }) {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <SessionWrapping>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          {children}
          <Toaster />
        </ThemeProvider>
      </SessionWrapping>
    </Suspense>
  )
}