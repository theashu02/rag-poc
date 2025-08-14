import { ChatInterface } from "./components/chatUI/ChatInterface";
import HealthCheck from "./components/common/HealthCheck";
import NetworkHealthBar from "./components/common/NetworkStatus";

export default function Home() {
  return (
    <div className="flex justify-center items-center mx-auto h-screen bg-blue-500">
      <NetworkHealthBar />
      <HealthCheck />
      <div className="flex flex-col items-center gap-4 w-screen bg-red-300">
        {/* <Chat /> */}
        <ChatInterface />
      </div>
    </div>
  );
}
