'use client'
import React, { useState, useEffect } from "react";
import { Wifi, WifiOff, Signal, Smartphone, Monitor } from "lucide-react";

interface NetworkHealth {
  ping: number | null;
  status: "excellent" | "good" | "fair" | "poor" | "offline";
  isOnline: boolean;
  connectionType: string;
  effectiveType: string;
  downlink: number | null;
  rtt: number | null;
  saveData: boolean;
}

// Extend Navigator interface for Network Information API
declare global {
  interface Navigator {
    connection?: {
      effectiveType: "4g" | "3g" | "2g" | "slow-2g";
      type:
        | "bluetooth"
        | "cellular"
        | "ethernet"
        | "none"
        | "wifi"
        | "wimax"
        | "other"
        | "unknown";
      downlink: number;
      rtt: number;
      saveData: boolean;
      addEventListener: (type: string, listener: EventListener) => void;
      removeEventListener: (type: string, listener: EventListener) => void;
    };
    mozConnection?: any;
    webkitConnection?: any;
  }
}

const NetworkHealthBar: React.FC = () => {
  const [health, setHealth] = useState<NetworkHealth>({
    ping: null,
    status: "offline",
    isOnline: false,
    connectionType: "unknown",
    effectiveType: "unknown",
    downlink: null,
    rtt: null,
    saveData: false,
  });

  // Get connection info using Network Information API
  const getConnectionInfo = () => {
    const connection =
      navigator.connection ||
      navigator.mozConnection ||
      navigator.webkitConnection;

    if (connection) {
      return {
        connectionType: connection.type || "unknown",
        effectiveType: connection.effectiveType || "unknown",
        downlink: connection.downlink || null,
        rtt: connection.rtt || null,
        saveData: connection.saveData || false,
      };
    }

    return {
      connectionType: "unknown",
      effectiveType: "unknown",
      downlink: null,
      rtt: null,
      saveData: false,
    };
  };

  // More accurate ping measurement using multiple endpoints
  const measureRealPing = async (): Promise<number | null> => {
    if (!navigator.onLine) return null;

    const endpoints = [
      "https://www.google.com/generate_204",
      "https://www.cloudflare.com/cdn-cgi/trace",
      "https://httpbin.org/status/200",
    ];

    const measurements: number[] = [];

    for (const endpoint of endpoints) {
      try {
        const startTime = performance.now();

        // Use fetch with AbortController for timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);

        await fetch(endpoint, {
          method: "HEAD",
          mode: "no-cors",
          cache: "no-cache",
          signal: controller.signal,
        });

        clearTimeout(timeoutId);
        const endTime = performance.now();
        measurements.push(Math.round(endTime - startTime));

        // Break early if we get a good measurement
        if (measurements.length >= 1) break;
      } catch (error) {
        // Continue to next endpoint
        continue;
      }
    }

    if (measurements.length === 0) return null;

    // Return average of measurements
    return Math.round(
      measurements.reduce((a, b) => a + b, 0) / measurements.length
    );
  };

  // Determine status based on multiple factors
  const determineNetworkStatus = (
    ping: number | null,
    rtt: number | null,
    effectiveType: string,
    isOnline: boolean
  ): NetworkHealth["status"] => {
    if (!isOnline) return "offline";

    // Use Network Information API RTT if available (more accurate)
    const latency = rtt || ping;

    if (!latency) {
      // Fallback to effective type if no latency data
      switch (effectiveType) {
        case "4g":
          return "excellent";
        case "3g":
          return "good";
        case "2g":
          return "fair";
        case "slow-2g":
          return "poor";
        default:
          return "fair";
      }
    }

    // Determine status based on latency
    if (latency < 50) return "excellent";
    if (latency < 100) return "good";
    if (latency < 200) return "fair";
    return "poor";
  };

  const getStatusColor = (status: NetworkHealth["status"]): string => {
    switch (status) {
      case "excellent":
        return "bg-emerald-500";
      case "good":
        return "bg-green-500";
      case "fair":
        return "bg-yellow-500";
      case "poor":
        return "bg-orange-500";
      case "offline":
        return "bg-red-500";
      default:
        return "bg-gray-500";
    }
  };

  const getConnectionIcon = (connectionType: string, effectiveType: string) => {
    switch (connectionType) {
      case "cellular":
        return <Smartphone className="w-3 h-3" />;
      case "wifi":
        return <Wifi className="w-3 h-3" />;
      case "ethernet":
        return <Monitor className="w-3 h-3" />;
      default:
        if (effectiveType === "4g" || effectiveType === "3g") {
          return <Signal className="w-3 h-3" />;
        }
        return <Wifi className="w-3 h-3" />;
    }
  };

  const formatSpeed = (downlink: number | null): string => {
    if (!downlink) return "";
    if (downlink >= 1) return `${downlink.toFixed(1)}Mbps`;
    return `${(downlink * 1000).toFixed(0)}Kbps`;
  };

  useEffect(() => {
    const updateNetworkHealth = async () => {
      const isOnline = navigator.onLine;
      const connectionInfo = getConnectionInfo();

      if (!isOnline) {
        setHealth({
          ping: null,
          status: "offline",
          isOnline: false,
          ...connectionInfo,
        });
        return;
      }

      // Get ping measurement
      const ping = await measureRealPing();

      // Determine status using all available data
      const status = determineNetworkStatus(
        ping,
        connectionInfo.rtt,
        connectionInfo.effectiveType,
        isOnline
      );

      setHealth({
        ping,
        status,
        isOnline,
        ...connectionInfo,
      });
    };

    // Initial check
    updateNetworkHealth();

    // Set up periodic checks (less frequent to avoid spam)
    const interval = setInterval(updateNetworkHealth, 5000);

    // Listen for network events
    const handleOnline = () => updateNetworkHealth();
    const handleOffline = () => {
      const connectionInfo = getConnectionInfo();
      setHealth((prev) => ({
        ...prev,
        ping: null,
        status: "offline",
        isOnline: false,
        ...connectionInfo,
      }));
    };

    // Listen for connection changes
    const handleConnectionChange = () => updateNetworkHealth();

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    const connection =
      navigator.connection ||
      navigator.mozConnection ||
      navigator.webkitConnection;
    if (connection) {
      connection.addEventListener("change", handleConnectionChange);
    }

    return () => {
      clearInterval(interval);
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
      if (connection) {
        connection.removeEventListener("change", handleConnectionChange);
      }
    };
  }, []);

  return (
    <div className="absolute top-5 right-6 z-50">
      <div className="flex items-center gap-1.5 bg-white/95 backdrop-blur-sm border border-gray-200/80 rounded-full px-2.5 py-1 shadow-lg text-xs font-medium">
        {/* Status indicator */}
        <div
          className={`w-1.5 h-1.5 rounded-full ${getStatusColor(health.status)} ${health.isOnline ? "animate-pulse" : ""}`}
        />

        {/* Connection icon */}
        <div className={health.isOnline ? "text-gray-600" : "text-red-500"}>
          {health.isOnline ? (
            getConnectionIcon(health.connectionType, health.effectiveType)
          ) : (
            <WifiOff className="w-3 h-3" />
          )}
        </div>

        {/* Network info */}
        <div className="flex items-center gap-1 text-gray-700">
          {health.isOnline ? (
            <>
              {/* Show RTT from Network API or measured ping */}
              <span className="font-mono">
                {health.rtt
                  ? `${health.rtt}ms`
                  : health.ping
                    ? `${health.ping}ms`
                    : "---"}
              </span>

              {/* Show connection type */}
              {health.effectiveType !== "unknown" && (
                <span className="text-gray-500 uppercase text-[10px]">
                  {health.effectiveType}
                </span>
              )}

              {/* Show speed if available */}
              {health.downlink && (
                <span className="text-gray-500 text-[10px]">
                  {formatSpeed(health.downlink)}
                </span>
              )}

              {/* Data saver indicator */}
              {health.saveData && (
                <span className="text-orange-500 text-[10px]">💾</span>
              )}
            </>
          ) : (
            <span className="text-red-500">Offline</span>
          )}
        </div>
      </div>
    </div>
  );
};

export default NetworkHealthBar;
