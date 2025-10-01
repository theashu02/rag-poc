import { toast } from "sonner"
import { CheckCircle, XCircle, AlertTriangle, Info } from "lucide-react"

export const modernToast = {
  success: (message: string) => 
    toast.success(message, { icon: <CheckCircle className="h-4 w-4" /> }),
  
  error: (message: string) => 
    toast.error(message, { icon: <XCircle className="h-4 w-4" /> }),
  
  warning: (message: string) => 
    toast.warning(message, { icon: <AlertTriangle className="h-4 w-4" /> }),
  
  info: (message: string) => 
    toast.info(message, { icon: <Info className="h-4 w-4" /> }),
  
  glass: (message: string) => 
    toast(message, { className: "toast-glass" }),
}
