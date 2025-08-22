'use client'

import { ReactNode, Suspense } from "react"
import { ThemeProvider } from "@/components/ThemeFiles/theme-provider"
import SessionWrapping from "@/lib/Auth/sessionWrapping"
import { Toaster } from "@/components/ui/sonner"
import InitUser from "@/app/components/common/InitUser"

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
          <InitUser />
          {children}
          <Toaster />
        </ThemeProvider>
      </SessionWrapping>
    </Suspense>
  )
}