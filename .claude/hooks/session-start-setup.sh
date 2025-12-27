#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up .NET environment...${NC}"

# Install .NET 10.0 if not already installed
if ! command -v dotnet &> /dev/null; then
  echo "Installing .NET SDK 10.0..."
  wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh
  chmod +x ./dotnet-install.sh
  ./dotnet-install.sh --channel 10.0
  rm dotnet-install.sh
else
  DOTNET_VERSION=$(dotnet --version)
  echo "Existing .NET version: $DOTNET_VERSION"
fi

# Set environment variables using CLAUDE_ENV_FILE (only available in SessionStart)
if [ -n "$CLAUDE_ENV_FILE" ]; then
  # Add dotnet to PATH
  echo 'export PATH="$HOME/.dotnet:$PATH"' >> "$CLAUDE_ENV_FILE"
  # Optional: Disable telemetry
  echo 'export DOTNET_CLI_TELEMETRY_OPTOUT=1' >> "$CLAUDE_ENV_FILE"
fi

# Restore project dependencies
echo -e "${YELLOW}Restoring .NET dependencies...${NC}"
cd "$CLAUDE_PROJECT_DIR"

# Find and restore all .csproj files
find . -name "*.csproj" -type f | while read -r csproj; do
  echo "Restoring: $csproj"
  dotnet restore "$csproj"
done

echo -e "${GREEN}.NET setup complete!${NC}"
exit 0
