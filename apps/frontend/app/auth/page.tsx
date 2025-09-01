import React from 'react'
import NetworkHealthBar from '../components/common/NetworkStatus'
import { LoginPage } from '../components/Auth/Login'

export default function page() {
  
  return (
    <div className="flex h-screen w-screen">
      <NetworkHealthBar />
      <LoginPage />
    </div>
  )
}
