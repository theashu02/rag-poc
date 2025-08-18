import { ChatInterface } from "./components/chatUI/ChatInterface";
import HealthCheck from "./components/common/HealthCheck";
import NetworkHealthBar from "./components/common/NetworkStatus";
import { Sidebar } from "./components/common/Sidebar";

export default function Home() {
  return (
    <div className="flex h-screen">
      {/* <Sidebar /> */}
      <NetworkHealthBar />
      <HealthCheck />
      <div className="flex flex-col items-center w-full bg-red-300">
        <ChatInterface />
      </div>
    </div>
  );
}
