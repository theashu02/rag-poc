'use client'

import NetworkHealthBar from "./components/common/NetworkStatus"
import { LoginPage } from "./components/Auth/Login";

export default function Home() {
  return (
    <div className="flex h-screen w-screen">
      <NetworkHealthBar />
      <LoginPage />
    </div>
  )
}